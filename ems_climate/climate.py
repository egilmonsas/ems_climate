from typing import Tuple

import pandas as pd
import requests

ELEMENTS_MINIMAL = "sum(precipitation_amount P1D), mean(air_temperature P1D), mean(air_pressure_at_sea_level P1D)"  # ", mean(wind_speed P1D), max(relative_humidity P1D)") #INPUT_insert additional elements
ELEMENTS = "sum(precipitation_amount P1D), mean(surface_downwelling_shortwave_flux_in_air PT1H), mean(wind_speed P1D), mean(air_temperature P1D), min(air_temperature P1D), max(air_temperature P1D), min(relative_humidity P1D),max(relative_humidity P1D), mean(air_pressure_at_sea_level P1D)"  # ", mean(wind_speed P1D), max(relative_humidity P1D)") #INPUT_insert additional elements
ELEMENTS_TEMP = "mean(air_temperature P1H)"  # ", mean(wind_speed P1D), max(relative_humidity P1D)") #INPUT_insert additional elements


class ClimateConnection:
    @staticmethod
    def get_climate_data(
        api_key: str,
        hydrological_station: str = "SN18700",
        time_resolution=None,
        date_range: Tuple[str, str] = ("2024-01-01", "2025-01-01"),
        elements: str = ELEMENTS_MINIMAL,
    ) -> pd.DataFrame:
        """
        api_key: str -> met.no api key (https://frost.met.no/howto.html)
        hydrological_station: str -> norwegian metrological station
        elements: str -> what to include, default "sum(precipitation_amount P1D), mean(surface_downwelling_shortwave_flux_in_air PT1H), mean(wind_speed P1D), mean(air_temperature P1D), min(air_temperature P1D), max(air_temperature P1D), min(relative_humidity P1D),max(relative_humidity P1D), mean(air_pressure_at_sea_level P1D)"
        date_range tuple(start_time,stoptime): strf format %Y-%m-%d
        """

        # Define endpoint and parameters
        endpoint = "https://frost.met.no/observations/v0.jsonld"
        parameters = {
            "sources": hydrological_station,
            "elements": elements,
            "referencetime": f"{date_range[0]}/{date_range[1]}",
        }

        if time_resolution:
            parameters["timeResolution"] = time_resolution
        # Issue an HTTP GET request
        r = requests.get(endpoint, parameters, auth=(api_key, ""))
        # Extract JSON data
        json = r.json()

        # Check if the request worked, print out any errors
        if r.status_code == 200:
            data = json["data"]

        else:
            print("Error! Returned status code %s" % r.status_code)
            print("Message: %s" % json["error"]["message"])
            print("Reason: %s" % json["error"]["reason"])

        # This will return a Dataframe with all of the observations in a table format
        array = []
        for i in range(len(data)):
            for observation in data[i]["observations"]:
                row = observation
                row["referenceTime"] = data[i]["referenceTime"]
                row["sourceId"] = data[i]["sourceId"]
                array.append(row)

        df = pd.DataFrame(array)
        df = df.set_index("referenceTime")
        df.index = pd.to_datetime(df.index)

        # resampling to daily values

        # precipitation
        df_prec = df[df["elementId"] == "sum(precipitation_amount P1D)"]
        df_prec = (df_prec[df_prec["timeOffset"] == "PT6H"])["value"]

        df_solar_radiation = df[
            df["elementId"] == "mean(surface_downwelling_shortwave_flux_in_air PT1H)"
        ]
        # solar radiation:
        df_solar_radiation = df_solar_radiation["value"].resample("D").mean()

        # temperature daily average
        df_temp_mean = (
            df[
                (df["elementId"] == "mean(air_temperature P1D)")
                & (df["timeOffset"] == "PT0H")
            ]
        )["value"]
        df_temp_min = (
            df[
                (df["elementId"] == "min(air_temperature P1D)")
                & (df["timeOffset"] == "PT0H")
            ]
        )["value"]
        df_temp_max = (
            df[
                (df["elementId"] == "max(air_temperature P1D)")
                & (df["timeOffset"] == "PT0H")
            ]
        )["value"]

        # humidity daily average
        df_hum_min = (df[df["elementId"] == "min(relative_humidity P1D)"])["value"]
        df_hum_max = (df[df["elementId"] == "max(relative_humidity P1D)"])["value"]

        # air_pressure daily average
        df_atm = (df[df["elementId"] == "mean(air_pressure_at_sea_level P1D)"])[
            "value"
        ]  # .resample("D").mean()

        # wind daily average
        df_wind = (df[df["elementId"] == "mean(wind_speed P1D)"])[
            "value"
        ]  # .resample("D").mean()

        df_met = pd.concat(
            [
                df_prec,
                df_solar_radiation,
                df_temp_mean,
                df_temp_min,
                df_temp_max,
                df_hum_min,
                df_hum_max,
                df_atm,
                df_wind,
            ],
            axis=1,
            keys=[
                "precipitation_sum",
                "solar_avg",
                "temp_mean",
                "temp_min",
                "temp_max",
                "hum_min",
                "hum_max",
                "atm_avg",
                "wind_avg",
            ],
        )
        df_met["solar_avg"] = df_met["solar_avg"]
        df_met.index = pd.to_datetime(df_met.index, format="%Y-%m-%d").tz_localize(None)

        return df_met

    @staticmethod
    def derive_penman_monteith_evapotranspiration(df_met: pd.DataFrame) -> pd.DataFrame:
        def compute_snowfall(row):
            if row["temp_mean"] <= 0:
                row["snowfall"] = row["precipitation_sum"]
                row["melt"] = 0
                row["frost"] = row["temp_mean"]
            else:
                row["snowfall"] = 0
                row["melt"] = -row["temp_mean"]
                row["frost"] = 0

            return row

        import math

        import numpy as np
        import pandas as pd

        # *****************************
        # Implement snow model correction
        # *****************************
        df_met = df_met.apply(compute_snowfall, axis=1)
        df_met["diff"] = df_met["snowfall"] + df_met["melt"]

        df_met = df_met.reset_index()
        df_met.loc[0, "snowdepth"] = 0

        for i in list(range(1, len(df_met))):
            if df_met.loc[i - 1, "snowdepth"] == 0:
                df_met.loc[i, "snowdepth"] = df_met.loc[i, "snowfall"]
            elif df_met.loc[i - 1, "snowdepth"] > 0:
                if df_met.loc[i - 1, "snowdepth"] > df_met.loc[i, "melt"] * -1:
                    df_met.loc[i, "snowdepth"] = (
                        df_met.loc[i - 1, "snowdepth"] + df_met.loc[i, "diff"]
                    )

                else:
                    df_met.loc[i, "snowdepth"] = 0
        df_met = df_met.set_index("referenceTime")
        df_met.index = pd.to_datetime(df_met.index, format="%Y-%m-%d").tz_localize(None)
        df_met["delta_snow"] = df_met["snowdepth"] - (
            df_met["snowdepth"].shift(1)
        ).fillna(0)
        df_met["tot_water"] = df_met["precipitation_sum"] - df_met["delta_snow"]
        # %%
        # ************************
        # Calculate estimated evaporation with Penman Monteith equation:
        # (metode): https://www.agraria.unirc.it/documentazione/materiale_didattico/1462_2016_412_24509.pdf
        # ***********************

        # Step 1. Mean daily temperature calculation
        df_met["mean_t"] = (df_met["temp_max"] + df_met["temp_min"]) / 2

        # Step 2. Mean daily solar radiation
        df_met["Rs"] = (
            df_met["solar_avg"] * 60 * 60 * 24 / 1000000
        )  # (solar_avg gitt i W/m2, ganger med 60*60*24 for å få watt pr. døgn, deler på 1*10^6 for å konvertere til MJ/døgnm2)

        # Step 3. Average wind speed, konvertering til 2m over bakken
        from numpy import log as ln

        df_met["u2"] = df_met["wind_avg"] * (4.87 / (ln((67.8 * 10) - 5.42)))

        # Step 4. Slope saturation water pressure
        exp = 2.7183
        df_met["slope"] = (
            4098
            * (0.6108 * exp * ((17.27 * df_met["mean_t"]) / (df_met["mean_t"] + 237.3)))
        ) / ((df_met["mean_t"] + 273.3) * (df_met["mean_t"] + 273.3))

        # Step 5. Atmospheric pressure pressure
        df_met["P"] = df_met["atm_avg"]

        # Step 6. Psycometric constant
        cp = 1.013 / 1000
        epsylon = 2.45
        lamdda = 0.622

        psycmetric_ratio = cp / (epsylon * lamdda)

        df_met["gamma"] = df_met["P"] * psycmetric_ratio

        # Step 7: The Delta term
        df_met["DT"] = df_met["slope"] / (
            df_met["slope"] + (df_met["gamma"] * (1 + 0.34 * df_met["u2"]))
        )

        # Step 8. Psi term (PT), the wind term
        df_met["PT"] = df_met["gamma"] / (
            df_met["slope"] + (df_met["gamma"] * (1 + 0.34 * df_met["u2"]))
        )

        # Step 9. Temperature term (TT)
        df_met["TT"] = (900 / (df_met["mean_t"] + 273)) * df_met["u2"]

        # Step 10 Mean saturation vapor pressure derived from air temperature
        df_met["eT_min"] = (
            0.6108 * exp * ((17.27 * df_met["temp_min"]) / (df_met["temp_min"] + 237.3))
        )
        df_met["eT_max"] = (
            0.6108 * exp * ((17.27 * df_met["temp_max"]) / (df_met["temp_max"] + 237.3))
        )

        df_met["es"] = (df_met["eT_max"] + df_met["eT_min"]) / 2

        # Step 11. Actual vapor pressure ea derived from relative humidity
        df_met["ea"] = (
            (
                (df_met["eT_min"] * (df_met["hum_max"] / 100))
                + (df_met["eT_max"] * (df_met["hum_min"] / 100))
            )
            / 2
        ).clip(
            lower=0
        )  # luker bort verider lavere enn 0 (får ikke negativt trykk)

        # Step 12.
        # The inverse relative distance earth-sun and solar declination
        df_met.index = pd.to_datetime(df_met.index)
        df_met["day_of_year"] = df_met.index.dayofyear
        df_met["dr"] = 1 + (
            0.033 * np.cos(df_met["day_of_year"] * ((2 * (math.pi) / 365)))
        )
        df_met["sol_dec"] = 0.0409 * np.sin(
            (df_met["day_of_year"] * ((2 * (math.pi) / 365))) - 1.39
        )

        # Step 13. Conversion of latitude in degrees to radians
        Oslo_lat_deg = 59.911491
        Oslo_lat_rad = ((math.pi) / 180) * Oslo_lat_deg

        # Step 14. Sunset hour angle
        df_met["ws"] = np.arccos(-np.tan(Oslo_lat_rad) * np.tan(df_met["sol_dec"]))

        # Step 15. Extraterrestian radiation Ra
        df_met["Ra"] = (((24 * 60) / (math.pi)) * 0.0820 * df_met["dr"]) * (
            (df_met["ws"] * np.sin(Oslo_lat_rad) * np.sin(df_met["sol_dec"]))
            + (np.cos(Oslo_lat_rad) * np.cos(df_met["sol_dec"]) * np.sin(df_met["ws"]))
        )

        # Step 16. Clear sky solar radiation
        z = 8
        df_met["Rso"] = df_met["Ra"] * (0.75 + (z / 100000))

        # Step 17. Net solar or net shortwave radiation Rns
        alpha_1 = 0.23  # canopy reflection coefficient for grass
        df_met["Rns"] = (1 - alpha_1) * df_met["Rs"]

        # Step 18. Net outgoing long wave solar radiation Rnl
        df_met["Rnl"] = (
            (4.903 / 1000000000)
            * (
                (
                    ((df_met["temp_max"] + 273.16) ** 4)
                    + ((df_met["temp_min"] + 273.16) ** 4)
                )
                / 2
            )
            * (0.34 - (0.14 * np.sqrt(df_met["ea"])))
            * ((1.35 * (df_met["Rs"] / df_met["Rso"])) - 0.35)
        )

        # Step 19. Net radiation
        df_met["Rn"] = df_met["Rns"] - df_met["Rnl"]  # (in MJ/m2day)
        df_met["Rng"] = df_met["Rn"] * 0.408  # (in mm)

        # Final step: Overall ETo equation

        # A. ETrad radiation term
        df_met["ET_rad"] = df_met["DT"] * df_met["Rng"]

        # B. ETwind wind term
        df_met["ET_wind"] = df_met["PT"] * (
            df_met["TT"] * (df_met["es"] - df_met["ea"])
        )

        # Final reference evaporation
        df_met["ET0"] = (df_met["ET_rad"] + df_met["ET_wind"]).clip(
            lower=0
        )  # luke ut negative verdier (gir ikke mening)

        # Final meteorology input data series:
        df_met["frostmengde"] = (df_met["temp_mean"] * 24).clip(upper=0)
        return df_met

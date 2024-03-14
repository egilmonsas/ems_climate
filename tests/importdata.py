import pytest
from ems_climate.climate import ClimateConnection


def test_climate():
    df_climate = ClimateConnection.get_climate_data(
        "efeaed67-a698-4982-a1c7-67ef9b2cab0e", date_range=("2023-01-01", "2025-01-01")
    )
    print(df_climate.head())
    assert len(df_climate.head()) > 0

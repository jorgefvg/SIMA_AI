import pandas as pd
from src.data_download import download_ashrae_data


def test_download_ashrae_data():
    df_meta, df_measurements = download_ashrae_data()

    assert isinstance(df_meta, pd.DataFrame)
    assert isinstance(df_measurements, pd.DataFrame)

    assert not df_meta.empty
    assert not df_measurements.empty

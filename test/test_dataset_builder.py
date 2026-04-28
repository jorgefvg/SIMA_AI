import pandas as pd
from src.dataset_builder import build_dataset


def test_build_dataset():
    df_meta = pd.DataFrame(
        {
            "building_id": [1],
            "region": ["a"],
            "building_type": ["office"],
            "cooling_type": ["air conditioned"],
            "records": [10],
        }
    )

    df_measurements = pd.DataFrame(
        {
            "building_id": [1, 1],
            "ta": [24, 25],
            "thermal_sensation": [0, 1],
            "rh": [50, 55],
            "t_out_isd": [20, 21],
            "t_out": [None, None],
        }
    )

    df = build_dataset(df_meta, df_measurements)

    assert isinstance(df, pd.DataFrame)
    assert "neutral_temp" in df.columns

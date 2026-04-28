import pandas as pd
from src.carga_datos import cargar_dataset


def test_cargar_dataset():
    df = cargar_dataset("data/adaptive_comfort_dataset.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "neutral_temp" in df.columns

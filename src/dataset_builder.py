"""
Construcción del dataset final con temperatura neutral por edificio.

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2025.11.14 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

from typing import Callable
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


## Ejecuta regresión lineal para calcular temperatura neutral
# @param dblg Datos de un edificio específico
# @return float Temperatura neutral estimada o NaN
def _run_lm(bldg: pd.DataFrame) -> float:

    try:
        lm_result = smf.ols(formula="ta ~ thermal_sensation", data=bldg).fit()

        if lm_result.pvalues["Intercept"] < 0.05:
            return lm_result.params["Intercept"]
        return np.nan
    except Exception:
        return np.nan


## Construye el dataset final listo para ML
# @param df_meta Metadatos de la base de datos
# @param df_measurements Mediciones de la base de datos
# @return pd.DataFrame Dataset con temperatura neutral por edificio
def build_dataset(df_meta: pd.DataFrame, df_measurements: pd.DataFrame) -> pd.DataFrame:

    df = df_measurements.loc[
        (~df_measurements["ta"].isna())
        & (~df_measurements["thermal_sensation"].isna())
        & (~df_measurements["rh"].isna())
        & (~(df_measurements["t_out_isd"].isna()) | ~(df_measurements["t_out"].isna()))
    ].copy()

    df["t_out_combined"] = df["t_out_isd"].fillna(df["t_out"])
    df = df.drop(columns=["t_out_isd", "t_out"])

    df = df.merge(
        df_meta[["building_id", "region", "building_type", "cooling_type", "records"]],
        on="building_id",
        how="left",
    )

    df = df[df["building_type"] == "office"]

    df_models = (
        df.groupby("building_id").apply(_run_lm, include_groups=False).reset_index()
    )
    df_models.columns = ["building_id", "neutral_temp"]

    df_models = df_models.merge(
        df_meta[["building_id", "records", "cooling_type", "region"]],
        on="building_id",
        how="left",
    )

    df_models["t_out_mean"] = df.groupby("building_id")["t_out_combined"].mean().values

    df_models = df_models.dropna()

    return df_models

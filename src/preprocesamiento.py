"""
Funciones para preparar los datos del entrenamiento

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2026.02.01 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


## Prepara los datos para entrenamiento
# @param df DataFrame original
# @return tuple[] X, y, pesos, encoder entrenado
def preparar_datos(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OneHotEncoder]:

    encoder = OneHotEncoder(sparse_output=False)

    # Variables categóricas
    X_cat = encoder.fit_transform(df[["cooling_type", "region"]])

    # Variable numérica
    X_num = df[["t_out_mean"]].values

    # Unión
    X = np.hstack([X_num, X_cat]).astype(np.float32)

    y = df["neutral_temp"].values.astype(np.float32)

    pesos = df["records"].values

    return X, y, pesos, encoder


## Prepara un nuevo dato para predicción
# @return np.ndarray Vector listo para el modelo
def transformar_nuevo_dato(
    encoder: OneHotEncoder,
    t_out: float,
    cooling: str,
    region: str,
) -> np.ndarray:

    X_num = np.array([[t_out]])

    df_cat = pd.DataFrame([[cooling, region]], columns=["cooling_type", "region"])

    X_cat = encoder.transform(df_cat)

    return np.hstack([X_num, X_cat]).astype(np.float32)

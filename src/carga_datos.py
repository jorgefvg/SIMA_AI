"""
Cargar el dataset procesado de ASHRAE DB II

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2026.02.01 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

import pandas as pd


## Carga el dataset desde un archivo CSV
# @param ruta Ruta del archivo CSV
# @return pd.DataFrame DataFrame con los datos
def cargar_dataset(ruta: str) -> pd.DataFrame:

    df = pd.read_csv(ruta)
    return df

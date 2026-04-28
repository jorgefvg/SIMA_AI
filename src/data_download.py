"""
Funciones para descargar los datasets originales
de ASHRAE DB II.

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2025.11.14 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

from typing import Tuple
import pandas as pd


URL_META = "https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II/raw/master/v2.1.0/db_metadata.csv"
URL_MEASUREMENTS = "https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II/raw/master/v2.1.0/db_measurements_v2.1.0.csv.gz"


## Descarga los datos metadata y measurements.
# @return Tuple[pd.DataFrame, pd.DataFrame] DataFrames de metadata y measurement.
def download_ashrae_data() -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_meta = pd.read_csv(URL_META)
    df_measurements = pd.read_csv(URL_MEASUREMENTS, low_memory=False)
    return df_meta, df_measurements

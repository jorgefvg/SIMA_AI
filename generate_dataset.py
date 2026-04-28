"""
Script para descargar y generar el dataset procesado a partir de ASHRAE DB II.

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2025.11.14 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

from src.data_download import download_ashrae_data
from src.dataset_builder import build_dataset


## Orquesta la descarga y construcción del dataset
# @return None
def main() -> None:

    df_meta, df_measurements = download_ashrae_data()
    df_final = build_dataset(df_meta, df_measurements)

    df_final.to_csv("data/adaptive_comfort_dataset.csv", index=False)
    print("Dataset guardado en data/adaptive_comfort_dataset.csv")


if __name__ == "__main__":
    main()

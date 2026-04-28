"""
Definición del modelo de Machine Learning

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2026.02.01 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

from sklearn.ensemble import RandomForestRegressor


## Crea el modelo Random Forest optimizado
# @return RandomForestRegressor Modelo configurado
def crear_modelo() -> RandomForestRegressor:

    return RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=2,
        random_state=42,
    )

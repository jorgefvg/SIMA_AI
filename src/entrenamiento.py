"""
Entrenamiento del modelo

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2026.02.01 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

import numpy as np
from sklearn.model_selection import train_test_split


## Divide los datos en entrenamiento y prueba
# @return train_test_split
def dividir_datos(
    X: np.ndarray,
    y: np.ndarray,
    pesos: np.ndarray,
):

    return train_test_split(X, y, pesos, test_size=0.2, random_state=42)


## Entrena el modelo con pesos
# @return modelo
def entrenar_modelo(modelo, X_train, y_train, pesos):

    modelo.fit(X_train, y_train, sample_weight=pesos)
    return modelo

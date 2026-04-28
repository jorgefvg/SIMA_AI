"""
Evaluación del modelo

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2026.02.01 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


## Evalúa el modelo e imprime métricas
# @return y_pred
def evaluar_modelo(modelo, X_test, y_test, pesos):

    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmse_w = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=pesos))
    r2_w = r2_score(y_test, y_pred, sample_weight=pesos)

    print("\n=== RESULTADOS ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"RMSE (weighted): {rmse_w:.4f}")
    print(f"R2 (weighted): {r2_w:.4f}")

    return y_pred

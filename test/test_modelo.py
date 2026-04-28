from src.modelo import crear_modelo
from sklearn.ensemble import RandomForestRegressor


def test_crear_modelo():
    modelo = crear_modelo()

    assert isinstance(modelo, RandomForestRegressor)
    assert modelo.n_estimators == 50
    assert modelo.max_depth == 5

import numpy as np
from src.carga_datos import cargar_dataset
from src.preprocesamiento import preparar_datos, transformar_nuevo_dato


def test_preparar_datos():
    df = cargar_dataset("data/adaptive_comfort_dataset.csv")

    X, y, pesos, encoder = preparar_datos(df)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(pesos, np.ndarray)

    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == pesos.shape[0]


def test_transformar_nuevo_dato():
    df = cargar_dataset("data/adaptive_comfort_dataset.csv")

    X, y, pesos, encoder = preparar_datos(df)

    X_new = transformar_nuevo_dato(
        encoder,
        t_out=25.0,
        cooling="air conditioned",
        region="americas",
    )

    assert X_new.shape[1] == X.shape[1]

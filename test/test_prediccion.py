from src.carga_datos import cargar_dataset
from src.preprocesamiento import preparar_datos, transformar_nuevo_dato
from src.modelo import crear_modelo
from src.entrenamiento import dividir_datos, entrenar_modelo
from src.prediccion import predecir


def test_prediccion():
    df = cargar_dataset("data/adaptive_comfort_dataset.csv")

    X, y, pesos, encoder = preparar_datos(df)

    X_train, X_test, y_train, y_test, w_train, w_test = dividir_datos(X, y, pesos)

    modelo = crear_modelo()
    modelo = entrenar_modelo(modelo, X_train, y_train, w_train)

    X_new = transformar_nuevo_dato(
        encoder,
        t_out=25.0,
        cooling="air conditioned",
        region="americas",
    )

    pred = predecir(modelo, X_new)

    assert len(pred) == 1

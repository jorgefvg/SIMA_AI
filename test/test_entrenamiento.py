from src.carga_datos import cargar_dataset
from src.preprocesamiento import preparar_datos
from src.entrenamiento import dividir_datos, entrenar_modelo
from src.modelo import crear_modelo


def test_entrenamiento():
    df = cargar_dataset("data/adaptive_comfort_dataset.csv")

    X, y, pesos, _ = preparar_datos(df)

    X_train, X_test, y_train, y_test, w_train, w_test = dividir_datos(X, y, pesos)

    modelo = crear_modelo()

    modelo_entrenado = entrenar_modelo(modelo, X_train, y_train, w_train)

    assert hasattr(modelo_entrenado, "predict")

from src.carga_datos import cargar_dataset
from src.preprocesamiento import preparar_datos
from src.entrenamiento import dividir_datos, entrenar_modelo
from src.modelo import crear_modelo
from src.evaluacion import evaluar_modelo


def test_evaluacion():
    df = cargar_dataset("data/adaptive_comfort_dataset.csv")

    X, y, pesos, _ = preparar_datos(df)

    X_train, X_test, y_train, y_test, w_train, w_test = dividir_datos(X, y, pesos)

    modelo = crear_modelo()
    modelo = entrenar_modelo(modelo, X_train, y_train, w_train)

    y_pred = evaluar_modelo(modelo, X_test, y_test, w_test)

    assert len(y_pred) == len(y_test)

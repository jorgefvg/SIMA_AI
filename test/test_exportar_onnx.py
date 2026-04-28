import os
from src.carga_datos import cargar_dataset
from src.preprocesamiento import preparar_datos
from src.modelo import crear_modelo
from src.entrenamiento import dividir_datos, entrenar_modelo
from src.exportar_onnx import exportar_modelo_onnx


def test_exportar_onnx():
    df = cargar_dataset("data/adaptive_comfort_dataset.csv")

    X, y, pesos, _ = preparar_datos(df)

    X_train, X_test, y_train, y_test, w_train, w_test = dividir_datos(X, y, pesos)

    modelo = crear_modelo()
    modelo = entrenar_modelo(modelo, X_train, y_train, w_train)

    ruta = "test_model.onnx"
    exportar_modelo_onnx(modelo, X.shape[1], ruta)

    assert os.path.exists(ruta)

    os.remove(ruta)

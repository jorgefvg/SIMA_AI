"""
Pipeline completo del modelo de inteligencia artificial:
1. Cargar dataset procesado
2. Preprocesar
3. Entrenar
4. Evaluar
5. Exportar ONNX
6. Prediciones

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2026.02.01 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

from src.carga_datos import cargar_dataset
from src.preprocesamiento import preparar_datos, transformar_nuevo_dato
from src.modelo import crear_modelo
from src.entrenamiento import dividir_datos, entrenar_modelo
from src.evaluacion import evaluar_modelo
from src.exportar_onnx import exportar_modelo_onnx


## Funcion principal
# @return nada
def main():

    # 1. Cargar datos
    df = cargar_dataset("data/adaptive_comfort_dataset.csv")

    print("Filas:", len(df))

    # 2. Preprocesamiento
    X, y, pesos, encoder = preparar_datos(df)

    # 3. División
    X_train, X_test, y_train, y_test, w_train, w_test = dividir_datos(X, y, pesos)

    # 4. Modelo
    modelo = crear_modelo()

    # 5. Entrenamiento
    modelo = entrenar_modelo(modelo, X_train, y_train, w_train)

    # 6. Evaluación
    evaluar_modelo(modelo, X_test, y_test, w_test)

    # 7. Exportar ONNX
    exportar_modelo_onnx(modelo, X.shape[1], "modelo_confort.onnx")

    # 8. Prueba de predicción
    X_nuevo = transformar_nuevo_dato(
        encoder,
        t_out=28.5,
        cooling="air conditioned",
        region="americas",
    )

    pred = modelo.predict(X_nuevo)
    print("\nPredicción ejemplo:", pred[0])

    print("\n=== ORDEN DE CATEGORÍAS ===")
    print(encoder.categories_)


if __name__ == "__main__":
    main()


# Adaptive Comfort - Random Forest Model

## Requisitos

- Instalar miniconda
- Instalar pre-commit
- En vs code se puede instalar la extension para formatear codigo .py cada vez que guardas (Black Formatter de Microsoft)
- Python 3.10 recomendado.
- Instalar dependencias:
    ```
    pip install -r requirements.txt
    ```

## Paso 1 - Generar dataset
```
python generate_dataset.py
```
Genera:
data/adaptive_comfort_dataset.csv

## Paso 2 - Entrenar modelo y exportar ONNX
```
python main.py
```
Genera:
adaptive_comfort_rf.onnx

## Uso en STM32

El archivo modelo_confort.onnx puede ser importado directamente en STM32Cube.AI para generar el código embebido.
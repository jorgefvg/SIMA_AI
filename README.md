
# Modelo de confort termico adaptativo - Random Forest

## Requisitos

- Instalar miniconda
- Instalar pre-commit
- En vs code se puede instalar la extension para formatear codigo .py cada vez que guardas (Black Formatter de Microsoft)
- Python 3.10 recomendado.
- Crear y activar el entorno de trabajo con conda:
    ```
    conda env create -f environment.yml
    conda activate sima-ai
    ```
- Nota: Tambien se pueden instalar las dependencias apartir de un archivo .txt:
    ```
    pip install -r requirements.txt
    ```
- Después de clonar el repositorio usted debería ejecutar el siguiente comando:

    ```
    pre-commit install
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
modelo_confort.onnx

## Uso en STM32

El archivo modelo_confort.onnx puede ser importado directamente en STM32Cube.AI para generar el código embebido.

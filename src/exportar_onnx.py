"""
Exportación del modelo a ONNX para uso en STM32 Cube.AI

### Registro de cambios
| REV | YYYY.MM.DD | Autor           | Descripción de los cambios                      |
|-----|------------|-----------------|-------------------------------------------------|
|   1 | 2026.02.01 | Jorge Vásquez   | Versión inicial del archivo                     |
"""

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


## Exporta el modelo a formato ONNX
# @return nada
def exportar_modelo_onnx(modelo, num_features: int, ruta: str):

    initial_type = [("float_input", FloatTensorType([None, num_features]))]

    onnx_model = convert_sklearn(modelo, initial_types=initial_type)

    with open(ruta, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Modelo exportado a {ruta}")

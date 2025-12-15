import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np

class DummyCalibDataReader(CalibrationDataReader):
    def __init__(self):
        self.enum = iter([{'images': np.random.randn(1, 3, 224, 224).astype(np.float32)}])

    def get_next(self):
        return next(self.enum, None)

def quantize_model():
    input_model = "models/efficientNet-b0.onnx"
    output_model = "models/efficientnet-b0-int8.onnx"

    print("🔍 Loading ONNX model:", input_model)

    # Validate
    model = onnx.load(input_model)
    onnx.checker.check_model(model)
    print("✔ Model is valid")

    print("⚙️ Running QLinear (static) quantization...")

    dr = DummyCalibDataReader()
    quantize_static(
        input_model,
        output_model,
        dr,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )

    print(f"🎯 Saved QLinear INT8 model: {output_model}")

if __name__ == "__main__":
    quantize_model()


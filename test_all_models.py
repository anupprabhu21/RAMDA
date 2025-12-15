import onnxruntime as ort
import numpy as np
import os

models = [
    ("FP32", "models/efficientNet-b0.onnx"),
    ("INT8", "models/efficientnet-b0-int8.onnx"),
    ("Pruned", "models/mobilenet_v3_pruned.onnx")
]

# Try OpenVINO first (INT8 acceleration)
available_providers = ort.get_available_providers()
print("Available Execution Providers:", available_providers)

if "OpenVINOExecutionProvider" in available_providers:
    providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
else:
    print("⚠️ OpenVINO not found. INT8 model may fail on CPU Execution Provider")
    providers = ["CPUExecutionProvider"]

dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

def get_input_name(session):
    return session.get_inputs()[0].name

for name, path in models:
    print(f"\n🚀 Testing {name} model ({path})")

    if not os.path.exists(path):
        print(f"❌ Model not found: {path}")
        continue

    try:
        session = ort.InferenceSession(path, providers=providers)
        input_name = get_input_name(session)
        print(f"🔹 Using input tensor name: {input_name}")

        # Run inference
        result = session.run(None, {input_name: dummy})[0]

        print(f"✔ Output shape: {result.shape}")
        print(f"🔥 Sample values: {result[0][:5]}")
    except Exception as e:
        print(f"❌ Failed to run {name} model:")
        print(str(e))


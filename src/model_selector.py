"""
model_selector.py
Simple rule-based selector for minimal approach.
You can replace or augment with an ML model (e.g., RandomForest) later.
"""
import json

# Paths (relative)
MODEL_FP32_ONNX = "models/efficientNet-b0.onnx"
MODEL_INT8_ONNX = "models/efficientnet-b0-int8.onnx"
MODEL_PRUNED_ONNX = "models/mobilenet_v3_pruned.onnx"

# Also include OpenVINO IR equivalents (same base names in models/openvino/)
MODEL_FP32_OV = "models/openvino/efficientNet-b0.xml"
MODEL_INT8_OV = "models/openvino/efficientnet-b0-int8.xml"
MODEL_PRUNED_OV = "models/openvino/mobilenet_v3_pruned.xml"

def select_model(telemetry, mode="pytorch"):
    """
    Rule-based selection:
    - If CPU < 50% and RAM > 1000MB => use FP32
    - If CPU between 50-80% => use INT8
    - If CPU > 80% or RAM < 400MB => use PRUNED
    'mode' can be 'pytorch' (onnxruntime) or 'openvino' (OpenVINO IR)
    Returns a dict with chosen model path and reason.
    """
    cpu = telemetry["cpu_percent"]
    ram = telemetry["available_ram_mb"]

    reason = ""
    if cpu < 50 and ram > 1000:
        chosen = MODEL_FP32_OV if mode == "openvino" else MODEL_FP32_ONNX
        reason = "low load -> FP32"
    elif cpu < 80 and ram > 400:
        chosen = MODEL_INT8_OV if mode == "openvino" else MODEL_INT8_ONNX
        reason = "moderate load -> INT8"
    else:
        chosen = MODEL_PRUNED_OV if mode == "openvino" else MODEL_PRUNED_ONNX
        reason = "high load -> pruned"

    return {"model_path": chosen, "reason": reason}

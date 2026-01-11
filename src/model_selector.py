"""
model_selector.py

Resource-Aware Model Selection Module
"""

# -------------------------------
# Model Paths (Relative)
# -------------------------------

# ONNX models (ONNX Runtime)
MODEL_FP32_ONNX = "models/efficientNet-b0.onnx"
MODEL_INT8_ONNX = "models/efficientnet-b0-int8.onnx"
MODEL_PRUNED_ONNX = "models/mobilenet_v3_pruned.onnx"

# OpenVINO IR models
MODEL_FP32_OV = "models/openvino/efficientNet-b0.xml"
MODEL_INT8_OV = "models/openvino/efficientnet-b0-int8.xml"
MODEL_PRUNED_OV = "models/openvino/mobilenet_v3_pruned.xml"


def select_model(telemetry, mode="pytorch"):
    """
    Select the best model based on system resources
    """

    cpu = telemetry["cpu_percent"]
    ram = telemetry["available_ram_mb"]

    if cpu < 50 and ram > 1000:
        model_type = "FP32"
        model_path = MODEL_FP32_OV if mode == "openvino" else MODEL_FP32_ONNX
        reason = "Low CPU load and sufficient RAM -> FP32 selected"

    elif cpu < 80 and ram > 400:
        model_type = "INT8"
        model_path = MODEL_INT8_OV if mode == "openvino" else MODEL_INT8_ONNX
        reason = "Moderate system load -> INT8 selected"

    else:
        model_type = "PRUNED"
        model_path = MODEL_PRUNED_OV if mode == "openvino" else MODEL_PRUNED_ONNX
        reason = "High CPU load or low RAM -> PRUNED selected"

    return {
        "model_type": model_type,
        "model_path": model_path,
        "cpu_percent": cpu,
        "available_ram_mb": ram,
        "reason": reason
    }

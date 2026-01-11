"""
main.py
Top-level script to:
 - collect telemetry
 - decide which model to use (onnx or openvino)
 - run inference on selected model
 - log results to outputs/inference_result.json
"""

import argparse
import json
import os
import time

from src.resource_monitor import get_telemetry
from src.model_selector import select_model
from src.inference_pytorch import infer_onnx

try:
    from src.openvino_infer import infer_openvino
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

OUTPUT_JSON = "outputs/inference_result.json"
LOG_FILE = "outputs/logs.txt"

os.makedirs("outputs", exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.ctime()}: {msg}\n")

def run_once(image_path, mode="pytorch"):
    pipeline_start = time.time()

    # 1️⃣ Collect telemetry
    telemetry = get_telemetry()

    # 2️⃣ Model selection
    decision = select_model(telemetry, mode=mode)
    model_path = decision["model_path"]
    model_type = decision.get("model_type", "unknown")

    # 3️⃣ Inference
    if mode == "openvino":
        if not OPENVINO_AVAILABLE:
            raise RuntimeError(
                "OpenVINO is not installed. Install openvino-runtime or switch to pytorch mode."
            )
        result = infer_openvino(model_path, image_path)
    else:
        result = infer_onnx(model_path, image_path)

    # 4️⃣ Total pipeline latency
    pipeline_latency_ms = round((time.time() - pipeline_start) * 1000.0, 2)

    # 5️⃣ Final output JSON
    out = {
        "telemetry": telemetry,
        "decision": {
            "model_type": model_type,
            "model_path": model_path,
            "reason": decision["reason"],
            "backend": mode
        },
        "result": result,
        "pipeline_latency_ms": pipeline_latency_ms
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    log(json.dumps(out))
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="data/sample.png",
        help="path to input image"
    )
    parser.add_argument(
        "--mode",
        choices=["pytorch", "openvino"],
        default="pytorch",
        help="inference backend"
    )
    args = parser.parse_args()

    out = run_once(args.image, mode=args.mode)
    print("Output saved to", OUTPUT_JSON)
    print(json.dumps(out, indent=2))

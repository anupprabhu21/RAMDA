"""
main.py
Top-level script to:
 - collect telemetry
 - decide which model to use (onnx or openvino)
 - run inference on selected model
 - log results to outputs/inference_result.json
"""

import argparse, json, os
from src.resource_monitor import get_telemetry
from src.model_selector import select_model
from src.inference_pytorch import infer_onnx
from src.openvino_infer import infer_openvino
import time

OUTPUT_JSON = "outputs/inference_result.json"
LOG_FILE = "outputs/logs.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.ctime()}: {msg}\n")

def run_once(image_path, mode="pytorch"):
    telemetry = get_telemetry()
    decision = select_model(telemetry, mode=mode)
    model_path = decision["model_path"]

    # Choose inference backend
    if mode == "openvino":
        # expect .xml file
        result = infer_openvino(model_path, image_path)
    else:
        # onnxruntime (pytorch path)
        result = infer_onnx(model_path, image_path)

    out = {
        "telemetry": telemetry,
        "decision": decision,
        "result": result
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log(json.dumps(out))
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/sample.png", help="path to input image")
    parser.add_argument("--mode", choices=["pytorch", "openvino"], default="pytorch", help="inference backend")
    args = parser.parse_args()
    out = run_once(args.image, mode=args.mode)
    print("Output saved to", OUTPUT_JSON)
    print(json.dumps(out, indent=2))

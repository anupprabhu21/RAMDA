"""
openvino_infer.py
Performs inference using OpenVINO Runtime on CPU (or other devices).
Requires openvino-runtime installed.
"""

from openvino.runtime import Core
from PIL import Image
import numpy as np
import time
import os
import json
import urllib.request

core = Core()

# -------------------------------
# Load ImageNet labels
# -------------------------------
def get_imagenet_labels(label_file="imagenet_labels.txt"):
    if not os.path.exists(label_file):
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        urllib.request.urlretrieve(url, "labels.json")
        with open("labels.json") as f:
            labels = json.load(f)
        with open(label_file, "w") as f:
            for l in labels:
                f.write(l + "\n")
        os.remove("labels.json")

    with open(label_file) as f:
        return [l.strip() for l in f.readlines()]

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(img_path, input_size=224):
    img = Image.open(img_path).convert("RGB").resize((input_size, input_size))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)

# -------------------------------
# OpenVINO Inference
# -------------------------------
def infer_openvino(model_xml, img_path):
    labels = get_imagenet_labels()

    model = core.read_model(model=model_xml)
    compiled = core.compile_model(model=model, device_name="CPU")

    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    input_data = preprocess_image(img_path)

    start = time.time()
    result = compiled([input_data])[output_layer]
    latency_ms = (time.time() - start) * 1000

    pred_idx = int(np.argmax(result, axis=1)[0])
    confidence = float(np.max(result))
    pred_label = labels[pred_idx]

    return {
        "engine": "openvino",
        "model": model_xml,
        "prediction": pred_label,
        "confidence": confidence,
        "latency_ms": latency_ms
    }

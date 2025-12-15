"""
openvino_infer.py
Performs inference using OpenVINO Runtime on CPU (or other devices).
Requires openvino-runtime installed.
"""
from openvino.runtime import Core
from PIL import Image
import numpy as np
import time

core = Core()

def preprocess_image(img_path, input_size=224):
    img = Image.open(img_path).convert('RGB').resize((input_size, input_size))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def infer_openvino(model_xml, img_path):
    """
    model_xml: path to .xml OpenVINO IR file (weights in .bin)
    """
    model = core.read_model(model=model_xml)
    compiled = core.compile_model(model=model, device_name="CPU")
    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    input_data = preprocess_image(img_path)
    start = time.time()
    res = compiled([input_data])[output_layer]
    latency_ms = (time.time() - start) * 1000.0
    pred = int(np.argmax(res, axis=1)[0])
    conf = float(np.max(res))
    return {"prediction": pred, "confidence": conf, "latency_ms": latency_ms, "model": model_xml}

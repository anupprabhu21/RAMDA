import onnxruntime as ort
from PIL import Image
import numpy as np
import time
import os
import json
import urllib.request

# -------------------------------
# Global label cache (load once)
# -------------------------------
_LABELS = None


def get_imagenet_labels(label_file="imagenet_labels.txt"):
    global _LABELS

    if _LABELS is not None:
        return _LABELS

    if not os.path.exists(label_file):
        print(f"[INFO] '{label_file}' not found. Downloading...")
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        local_json = "imagenet_labels.json"

        urllib.request.urlretrieve(url, local_json)

        with open(local_json) as f:
            labels = json.load(f)

        with open(label_file, "w") as f:
            for label in labels:
                f.write(label + "\n")

        os.remove(local_json)
        print(f"[INFO] Labels saved to '{label_file}'")

    with open(label_file) as f:
        _LABELS = [line.strip() for line in f.readlines()]

    return _LABELS


# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(img_path, input_size=224):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((input_size, input_size))

    arr = np.array(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


# -------------------------------
# Softmax for confidence
# -------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


# -------------------------------
# PyTorch / ONNX Runtime Inference
# -------------------------------
def infer_onnx(model_path, image_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    labels = get_imagenet_labels()

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    input_tensor = preprocess_image(image_path)

    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    latency_ms = (time.time() - start_time) * 1000

    logits = outputs[0]
    probs = softmax(logits)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(probs[0][pred_idx])
    prediction = labels[pred_idx]

    return {
        "mode": "pytorch",
        "model_path": model_path,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2)
    }


# -------------------------------
# CLI support (keep this!)
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX Runtime Inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)

    args = parser.parse_args()

    output = infer_pytorch_onnx(args.model, args.image)
    print(json.dumps(output, indent=2))

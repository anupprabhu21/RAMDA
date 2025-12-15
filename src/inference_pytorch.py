import onnxruntime as ort
from PIL import Image
import numpy as np
import time
import os
import json
import urllib.request

# -------------------------------
# Function to ensure labels exist
# -------------------------------
def get_imagenet_labels(label_file="imagenet_labels.txt"):
    if not os.path.exists(label_file):
        print(f"[INFO] '{label_file}' not found. Downloading...")
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        local_json = "imagenet_labels.json"
        urllib.request.urlretrieve(url, local_json)
        with open(local_json) as f_json:
            labels = json.load(f_json)
        # Save as plain text
        with open(label_file, "w") as f_txt:
            for label in labels:
                f_txt.write(label + "\n")
        os.remove(local_json)
        print(f"[INFO] Saved labels to '{label_file}'")
    # Load labels
    with open(label_file) as f:
        return [line.strip() for line in f.readlines()]

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(img_path, input_size=224):
    img = Image.open(img_path).convert('RGB').resize((input_size, input_size))
    arr = np.array(img).astype(np.float32) / 255.0
    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

# -------------------------------
# ONNX Inference
# -------------------------------
def infer_onnx(model_path, img_path, input_name=None, label_file="imagenet_labels.txt"):
    # Load labels
    imagenet_labels = get_imagenet_labels(label_file)

    # Load ONNX model
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    if input_name is None:
        input_name = sess.get_inputs()[0].name

    # Preprocess image
    input_data = preprocess_image(img_path)

    # Run inference
    start = time.time()
    outputs = sess.run(None, {input_name: input_data})
    latency_ms = (time.time() - start) * 1000.0

    # Get predicted class index and confidence
    logits = outputs[0]
    pred_idx = int(np.argmax(logits, axis=1)[0])
    conf = float(np.max(logits))
    pred_label = imagenet_labels[pred_idx]

    # Return result in desired JSON format
    return {
        "result": {
            "prediction": pred_label,
            "confidence": conf,
            "latency_ms": latency_ms,
            "model": model_path
        }
    }

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX Inference")
    parser.add_argument("--model", type=str, default="models/efficientNet-b0.onnx", help="ONNX model path")
    parser.add_argument("--image", type=str, required=True, help="Image path")
    args = parser.parse_args()

    out = infer_onnx(args.model, args.image)
    print(json.dumps(out, indent=2))


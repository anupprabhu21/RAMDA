📌 Resource-Aware Model Deployment Agent on Edge Platforms

🔍 Overview

This project implements a Resource-Aware Model Deployment Agent that dynamically selects and executes the most suitable deep learning model variant on an edge device, based on real-time system resource availability.

Instead of using a single fixed model, the system intelligently switches between:

FP32 (high accuracy)

INT8 quantized (balanced)

Pruned / lightweight model (low resource usage)

This approach improves latency, efficiency, and adaptability, making it suitable for edge AI deployments.

🎯 Key Objectives

Enable adaptive AI inference on resource-constrained edge platforms

Compare accuracy vs performance trade-offs across optimized model variants

Demonstrate runtime decision logic driven by system telemetry

🧠 Models Used

| Model Variant             | Description                     | Purpose                         |
| ------------------------- | ------------------------------- | ------------------------------- |
| EfficientNet-B0 (FP32)    | Full-precision CNN              | High accuracy                   |
| EfficientNet-B0 (INT8)    | Quantized ONNX model            | Faster inference, smaller size  |
| MobileNetV3 (Lightweight) | Architecture-level pruned model | Low power / constrained devices |


Note: Pruning in this phase is architectural (lightweight network) rather than weight-level sparsity.

🏗️ System Architecture (High Level)

1. Input image is received
2. System metrics are collected (CPU, memory)
3. Deployment agent evaluates resource availability
4. Best model variant is selected
5. Inference is executed
6. Prediction and performance metrics are logged

📁 Directory Structure
resource-aware-edge-agent/
├── models/                 # ONNX models (generated locally)
│   ├── efficientNet-b0.onnx
│   ├── efficientnet-b0-int8.onnx
│   ├── mobilenet_v3_pruned.onnx
│   └── openvino/
├── data/
│   └── sample.jpg
├── src/
│   ├── resource_monitor.py
│   ├── model_selector.py
│   ├── convert_to_onnx.py
│   └── main.py
├── optimize_quantize.py
├── optimize_prune.py
├── test_all_models.py
├── requirements.txt
├── setup.sh
└── README.md


⚙️ Setup Instructions

1️⃣ Clone Repository

git clone https://github.com/<your-username>/resource-aware-edge-agent.git
cd resource-aware-edge-agent

2️⃣ One-Command Setup (Recommended)

./setup.sh


This will:

Create a virtual environment

Install required dependencies

Activate environment:

source venv/bin/activate

▶️ Running the Project
🔹 Test All Model Variants
python3 test_all_models.py


Expected:

FP32, INT8, and lightweight models run successfully

Output logits and shape (1, 1000) are printed

🔹 Run Full Pipeline
python3 src/main.py --image data/sample.jpg


Output:

Selected model

Prediction

Execution latency

Resource usage

📊 Results Summary
| Model  | Size    | Inference Speed | Accuracy    |
| ------ | ------- | --------------- | ----------- |
| FP32   | ~21 MB  | Slowest         | Highest     |
| INT8   | ~5.6 MB | Faster          | Slight drop |
| Pruned | ~21 MB  | Fastest         | Acceptable  |

🧪 Execution Backends

ONNX Runtime (CPU) – default & portable

OpenVINO (optional) – optimized for Intel edge devices

OpenVINO is optional and hardware-dependent.
The project remains functional without it.

🚀 Novel Contribution

Integrates system telemetry with model deployment

Demonstrates runtime-adaptive inference

Practical implementation of resource-aware AI

🔮 Future Work

Weight-level structured pruning

Reinforcement learning-based model selection

GPU / NPU acceleration

Real-time video inference support

🎓 Academic Note

This project is developed as part of an M.Tech (AI/ML) academic project with emphasis on edge AI deployment and system integration.

🧾 License

This project is intended for academic and research use.

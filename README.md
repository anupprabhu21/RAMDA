
# Resource-Aware Edge Model Deployment Agent

## Overview
This project demonstrates a resource-aware deployment agent that selects between multiple model variants (FP32, INT8, Pruned) at runtime based on device telemetry (CPU, RAM, temperature). It supports two inference backends:
- ONNXRuntime (CPU) — quick setup
- OpenVINO Runtime (Intel optimized) — recommended for Intel edge devices

## Directory Structure
resource-aware-edge-agent/
├── models/ # contains ONNX and OpenVINO IR files (not included)
├── data/
│ └── sample.jpg
├── src/
│ ├── resource_monitor.py
│ ├── model_selector.py
│ ├── inference_pytorch.py
│ ├── openvino_infer.py
│ ├── convert_to_onnx.py
│ └── main.py
├── outputs/
├── requirements.txt
└── README.md


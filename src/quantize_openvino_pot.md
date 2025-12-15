# quantize_openvino_pot.md (high-level steps)

1. Convert the FP32 ONNX -> OpenVINO IR:
   mo --input_model models/efficientnet_b0_fp32.onnx --output_dir models/openvino/

2. Create a data loader script for POT that yields validation images.

3. Run POT to produce INT8 IR:
   pot -c quantization_config.json

4. The output will be models/openvino/int8/...xml . Use those IR files in openvino_infer.py

import onnx
from onnxruntime_tools import optimizer
import os

INPUT_MODEL = "models/efficientNet-b0.onnx"
OUTPUT_MODEL = "models/mobilenet_v3_pruned.onnx"  # keeping your naming style

def prune_model():
    if not os.path.exists(INPUT_MODEL):
        print(f"❌ Model not found: {INPUT_MODEL}")
        return
    
    print("🔧 Loading model for pruning:", INPUT_MODEL)

    opt = optimizer.optimize_model(
        INPUT_MODEL,
        model_type='bert',  # although BERT flagged, optimization still applies to CNN graphs
        num_heads=0,
        hidden_size=0
    )

    opt.prune_graph()
    opt.save_model_to_file(OUTPUT_MODEL)

    print("✅ Pruned model saved →", OUTPUT_MODEL)


if __name__ == "__main__":
    prune_model()


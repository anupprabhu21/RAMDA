import streamlit as st
import json
import subprocess
import os
import time
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "outputs", "inference_result.json")

st.set_page_config(page_title="RAMDA – Edge AI Deployment Agent", layout="wide")

st.title("🧠 Resource-Aware Model Deployment Agent (RAMDA)")
st.markdown("**Edge AI system that dynamically selects FP32 / INT8 / PRUNED models based on system load.**")

# -----------------------------------
# Sidebar Controls
# -----------------------------------
st.sidebar.header("⚙️ Controls")

backend = st.sidebar.radio(
    "Inference Backend",
    ["pytorch", "openvino"]
)

image_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

run_button = st.sidebar.button("🚀 Run Inference")

# -----------------------------------
# Image Preview
# -----------------------------------
if image_file:
    img = Image.open(image_file)
    st.image(img, caption="Input Image", width=300)

# -----------------------------------
# Run Inference
# -----------------------------------
if run_button and image_file:
    with open("temp_input.png", "wb") as f:
        f.write(image_file.getbuffer())

    cmd = [
        "python3",
        "-m",
        "src.main",
        "--image",
        "temp_input.png",
        "--mode",
        backend
    ]

    st.info("Running deployment agent...")
    start = time.time()
    subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = round((time.time() - start) * 1000, 2)

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            data = json.load(f)

        # -----------------------------------
        # Telemetry
        # -----------------------------------
        st.subheader("📊 System Telemetry")
        col1, col2, col3 = st.columns(3)
        col1.metric("CPU Usage (%)", round(data["telemetry"]["cpu_percent"], 2))
        col2.metric("Available RAM (MB)", round(data["telemetry"]["available_ram_mb"], 2))
        col3.metric("Temperature (°C)", round(data["telemetry"]["temperature_c"], 2))

        # -----------------------------------
        # Decision
        # -----------------------------------
        st.subheader("🧩 Deployment Agent Decision")
        st.success(f"**Selected Model:** {data['decision']['model_type']}")
        st.write("**Reason:**", data["decision"]["reason"])
        st.write("**Backend:**", data["decision"]["backend"])

        # -----------------------------------
        # Inference Result
        # -----------------------------------
        st.subheader("🎯 Inference Output")
        st.write("**Prediction:**", data["result"]["prediction"])
        st.write("**Confidence:**", round(data["result"]["confidence"], 4))
        st.write("**Inference Latency (ms):**", round(data["result"]["latency_ms"], 2))
        st.write("**Pipeline Latency (ms):**", data["pipeline_latency_ms"])

        st.success(f"Completed in {elapsed} ms")

    else:
        st.error("Inference output not found!")

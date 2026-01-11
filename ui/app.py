import streamlit as st
import json
import subprocess
import os
import time
from PIL import Image
from collections import deque

# -----------------------------------
# Paths
# -----------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "outputs", "inference_result.json")

# -----------------------------------
# Streamlit Setup
# -----------------------------------
st.set_page_config(
    page_title="RAMDA – Edge AI Deployment Agent",
    layout="wide"
)

st.title("🧠 Resource-Aware Model Deployment Agent (RAMDA)")
st.markdown(
    "**Edge AI system that dynamically selects FP32 / INT8 / PRUNED models based on real-time system load.**"
)

# -----------------------------------
# Sidebar Controls
# -----------------------------------
st.sidebar.header("⚙️ Controls")

backend = st.sidebar.radio(
    "Inference Backend",
    ["pytorch", "openvino"]
)

image_file = st.sidebar.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg"]
)

auto_refresh = st.sidebar.checkbox("🔄 Continuous Monitoring", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 5, 2)

run_button = st.sidebar.button("🚀 Run Inference Now")

# -----------------------------------
# Session State (for live graphs)
# -----------------------------------
if "cpu_hist" not in st.session_state:
    st.session_state.cpu_hist = deque(maxlen=30)
    st.session_state.model_hist = deque(maxlen=30)

# -----------------------------------
# Image Preview
# -----------------------------------
if image_file:
    img = Image.open(image_file)
    st.image(img, caption="Input Image", width=300)

# -----------------------------------
# Save Uploaded Image
# -----------------------------------
def save_image(file):
    os.makedirs("data", exist_ok=True)
    path = "data/ui_input.png"
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path

# -----------------------------------
# Run Deployment Agent
# -----------------------------------
def run_agent(image_path, backend):
    cmd = [
        "python3",
        "-m",
        "src.main",
        "--image",
        image_path,
        "--mode",
        backend
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)

# -----------------------------------
# Trigger Inference
# -----------------------------------
should_run = run_button or auto_refresh

if should_run and image_file:
    image_path = save_image(image_file)
    run_agent(image_path, backend)

# -----------------------------------
# Load Output
# -----------------------------------
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON) as f:
        data = json.load(f)

    telemetry = data.get("telemetry", {})
    decision = data.get("decision", {})
    result = data.get("result", {})

    cpu = telemetry.get("cpu_percent", 0.0)
    ram = telemetry.get("available_ram_mb", 0.0)
    temp = telemetry.get("temperature_c", 0.0)
    model_type = decision.get("model_type", "Unknown")

    st.session_state.cpu_hist.append(cpu)
    st.session_state.model_hist.append(model_type)

    # -----------------------------------
    # Telemetry
    # -----------------------------------
    st.subheader("📊 Live System Telemetry")
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU Usage (%)", round(cpu, 2))
    col2.metric("Available RAM (MB)", round(ram, 2))
    col3.metric("Temperature (°C)", round(temp, 2))

    # -----------------------------------
    # Decision
    # -----------------------------------
    st.subheader("🧩 Deployment Agent Decision")
    st.success(f"**Selected Model:** {model_type}")
    st.write("**Reason:**", decision.get("reason", "N/A"))
    st.write("**Backend:**", decision.get("backend", backend))

    # -----------------------------------
    # Inference Result
    # -----------------------------------
    st.subheader("🎯 Inference Output")
    st.write("**Prediction:**", result.get("prediction", "N/A"))

    if "confidence" in result:
        st.write("**Confidence:**", round(result["confidence"], 4))

    if "latency_ms" in result:
        st.write("**Inference Latency (ms):**", round(result["latency_ms"], 2))

    st.write(
        "**Total Pipeline Latency (ms):**",
        data.get("total_pipeline_latency_ms", "N/A")
    )

    # -----------------------------------
    # Live Graph
    # -----------------------------------
    st.subheader("📈 CPU Load vs Time (Live)")
    st.line_chart(list(st.session_state.cpu_hist))

else:
    st.warning("No inference output found yet.")

# -----------------------------------
# Auto-refresh
# -----------------------------------
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

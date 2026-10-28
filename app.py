from pathlib import Path
import time
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from sensor_simulator import generate_sensor_reading

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "irrigation_model.joblib"
DATA_PATH = ROOT / "data" / "smart_agriculture_dataset.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Smart Agriculture | Azure ML + IoT Edge", page_icon="🌱", layout="wide")

st.title("🌱 Smart Agriculture: Azure ML + IoT Edge")
st.caption("Hardware-free mini project • Simulated sensors • Local edge inference")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run `python train_model.py` in the project folder, then refresh this page.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

if "history" not in st.session_state:
    st.session_state.history = []
if "latest" not in st.session_state:
    st.session_state.latest = generate_sensor_reading()

with st.sidebar:
    st.header("Sensor simulator")
    auto_refresh = st.toggle("Auto-refresh simulation", value=False)
    interval = st.slider("Refresh interval (seconds)", min_value=2, max_value=10, value=4)
    if st.button("Generate sensor reading", use_container_width=True):
        st.session_state.latest = generate_sensor_reading()
    st.markdown("---")
    st.markdown("**Architecture**")
    st.write("Python simulator → ML model → local edge prediction → dashboard")
    st.caption("Azure model registration is available as an optional step.")

reading = st.session_state.latest
input_df = pd.DataFrame([[reading[f] for f in features]], columns=features)
prediction = int(model.predict(input_df)[0])
probabilities = model.predict_proba(input_df)[0]
irrigation_probability = float(probabilities[list(model.classes_).index(1)]) if 1 in model.classes_ else float(prediction)

result = {
    **reading,
    "irrigation_needed": prediction,
    "irrigation_probability": round(irrigation_probability * 100, 1)
}
if not st.session_state.history or st.session_state.history[-1]["timestamp"] != reading["timestamp"]:
    st.session_state.history.append(result)
st.session_state.history = st.session_state.history[-100:]

if prediction == 1:
    st.warning(f"💧 Irrigation recommended — model confidence for 'needed': {irrigation_probability:.0%}")
else:
    st.success(f"✅ Irrigation not currently required — model confidence for 'needed': {irrigation_probability:.0%}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Soil moisture", f"{reading['soil_moisture']}%", help="Simulated sensor reading")
c2.metric("Temperature", f"{reading['temperature']} °C")
c3.metric("Humidity", f"{reading['humidity']}%")
c4.metric("Rainfall", f"{reading['rainfall_mm']} mm")

st.subheader("Latest prediction")
left, right = st.columns([1, 1])
with left:
    st.write("**Simulated sensor event**")
    st.json(reading)
    st.write(f"**Edge prediction:** {'Irrigation needed' if prediction else 'No irrigation needed'}")
    st.write(f"**Irrigation-needed probability:** {irrigation_probability:.1%}")
with right:
    gauge = pd.DataFrame({"State": ["Irrigation needed", "Other"], "Probability": [irrigation_probability, 1-irrigation_probability]})
    fig = px.pie(gauge, names="State", values="Probability", hole=0.65, title="Model probability")
    fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent sensor history")
history_df = pd.DataFrame(st.session_state.history)
if not history_df.empty:
    st.line_chart(history_df.set_index("timestamp")[["soil_moisture", "humidity"]])
    st.dataframe(history_df.tail(10), use_container_width=True, hide_index=True)
    st.download_button(
        "Download prediction history (CSV)",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="smart_agriculture_prediction_history.csv",
        mime="text/csv"
    )
else:
    st.info("Generate a sensor reading to start the history.")

if METRICS_PATH.exists():
    import json
    metrics = json.loads(METRICS_PATH.read_text())
    with st.expander("Model information and evaluation"):
        st.write(f"**Test accuracy:** {metrics['accuracy']:.1%}")
        st.write(f"**Training rows:** {metrics['train_rows']:,} | **Test rows:** {metrics['test_rows']:,}")
        st.write("**Features:** " + ", ".join(metrics["features"]))
        st.caption(metrics["note"])

st.caption("Educational simulation only. Synthetic data and predictions must not be used for real irrigation decisions.")

if auto_refresh:
    time.sleep(interval)
    st.session_state.latest = generate_sensor_reading()
    st.rerun()

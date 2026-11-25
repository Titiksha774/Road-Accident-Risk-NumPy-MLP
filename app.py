from pathlib import Path

import pandas as pd
import streamlit as st

from risk_model import predict_dataframe, risk_level


MODEL_PATH = Path("road_accident_mlp.pkl")


st.set_page_config(page_title="Road Accident Risk Prediction", page_icon="🚦", layout="centered")
st.title("Road Accident Risk Prediction")
st.caption("Deep Learning / MLP model for proactive road-condition risk estimation")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

left, right = st.columns(2)
with left:
    road_type = st.selectbox("Road type", ["urban", "rural", "highway"])
    num_lanes = st.slider("Number of lanes", 1, 6, 2)
    curvature = st.slider("Road curvature", 0.0, 1.0, 0.50, 0.01)
    speed_limit = st.slider("Speed limit", 20, 80, 45, 5)
    lighting = st.selectbox("Lighting", ["daylight", "dim", "night"])
    weather = st.selectbox("Weather", ["clear", "rainy", "foggy"])

with right:
    road_signs_present = st.checkbox("Road signs present", value=True)
    public_road = st.checkbox("Public road", value=True)
    time_of_day = st.selectbox("Time of day", ["morning", "afternoon", "evening", "night"])
    holiday = st.checkbox("Holiday")
    school_season = st.checkbox("School season", value=True)
    num_reported_accidents = st.slider("Previous reported accidents", 0, 10, 1)

input_df = pd.DataFrame(
    [
        {
            "road_type": road_type,
            "num_lanes": num_lanes,
            "curvature": curvature,
            "speed_limit": speed_limit,
            "lighting": lighting,
            "weather": weather,
            "road_signs_present": road_signs_present,
            "public_road": public_road,
            "time_of_day": time_of_day,
            "holiday": holiday,
            "school_season": school_season,
            "num_reported_accidents": num_reported_accidents,
        }
    ]
)

score = float(predict_dataframe(input_df, MODEL_PATH)[0])
level = risk_level(score)

st.metric("Predicted Accident Risk", f"{score:.3f}", level)
st.progress(score)
st.dataframe(input_df, use_container_width=True)

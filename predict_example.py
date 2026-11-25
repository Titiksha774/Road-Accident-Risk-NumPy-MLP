import pandas as pd

from risk_model import predict_dataframe, risk_level


example = pd.DataFrame(
    [
        {
            "road_type": "street",
            "num_lanes": 1,
            "curvature": 0.82,
            "speed_limit": 20,
            "lighting": "daylight",
            "weather": "cloudy",
            "road_signs_present": False,
            "public_road": True,
            "time_of_day": "evening",
            "holiday": False,
            "school_season": True,
            "num_reported_accidents": 3,
        }
    ]
)

score = float(predict_dataframe(example)[0])
print(f"Risk Score: {score:.3f}")
print(f"Risk Level: {risk_level(score)}")

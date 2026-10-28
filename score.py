
import json
import os
from pathlib import Path

import joblib
import pandas as pd

model = None


def init():
    global model

    model_dir = Path(os.environ["AZUREML_MODEL_DIR"])
    model_files = list(model_dir.rglob("*.joblib"))

    if not model_files:
        raise FileNotFoundError(
            f"No .joblib model file found in {model_dir}"
        )

    # Prefer the model trained for this project.
    model_path = next(
        (p for p in model_files if p.name == "irrigation_model.joblib"),
        model_files[0],
    )
    model = joblib.load(model_path)


def run(raw_data):
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)

    data = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or not data:
        return {"error": "Provide sensor readings as a JSON object or list."}

    # Preserve the feature order used during training.
    if data and isinstance(data[0], dict):
        frame = pd.DataFrame(data)
        if hasattr(model, "feature_names_in_"):
            frame = frame.reindex(columns=model.feature_names_in_)
        else:
            frame = frame[
                ["soil_moisture", "temperature", "humidity", "rainfall"]
            ]
    else:
        frame = pd.DataFrame(
            data,
            columns=["soil_moisture", "temperature", "humidity", "rainfall"],
        )

    predictions = model.predict(frame)
    probabilities = (
        model.predict_proba(frame)
        if hasattr(model, "predict_proba")
        else None
    )

    results = []
    for i, prediction in enumerate(predictions):
        result = {
            "predicted_class": prediction.item()
            if hasattr(prediction, "item") else prediction,
            "irrigation_needed": bool(prediction == 1),
        }

        if probabilities is not None:
            result["class_probabilities"] = probabilities[i].tolist()

        results.append(result)

    return {"predictions": results}

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["soil_moisture", "temperature", "humidity", "rainfall_mm"]

def generate_dataset(n=2500, seed=42):
    """Create synthetic classroom data with a transparent rule-derived target."""
    rng = np.random.default_rng(seed)
    soil = rng.uniform(5, 95, n)
    temp = rng.uniform(12, 43, n)
    humidity = rng.uniform(15, 100, n)
    rain = np.clip(rng.gamma(shape=1.4, scale=3.0, size=n), 0, 35)
    # Irrigation is more likely when soil is dry, it is warm, and recent rain is low.
    score = (
        (soil < 35).astype(int) * 2
        + (temp > 30).astype(int)
        + (humidity < 40).astype(int)
        - (rain >= 5).astype(int) * 2
    )
    labels = (score >= 2).astype(int)
    # Small label noise makes the task less perfectly deterministic.
    flip = rng.random(n) < 0.035
    labels[flip] = 1 - labels[flip]
    return pd.DataFrame({
        "soil_moisture": np.round(soil, 2),
        "temperature": np.round(temp, 2),
        "humidity": np.round(humidity, 2),
        "rainfall_mm": np.round(rain, 2),
        "irrigation_needed": labels
    })

def main():
    df = generate_dataset()
    csv_path = DATA_DIR / "smart_agriculture_dataset.csv"
    df.to_csv(csv_path, index=False)

    X = df[FEATURES]
    y = df["irrigation_needed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=180, max_depth=12, class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    model_path = MODEL_DIR / "irrigation_model.joblib"
    joblib.dump({"model": model, "features": FEATURES}, model_path)

    metrics = {
        "accuracy": round(accuracy, 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "dataset": str(csv_path.relative_to(ROOT)),
        "model": str(model_path.relative_to(ROOT)),
        "note": "Synthetic classroom dataset; not validated for real farming."
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Dataset saved: {csv_path}")
    print(f"Model saved:   {model_path}")
    print(f"Test accuracy: {accuracy:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Training complete.")

if __name__ == "__main__":
    main()

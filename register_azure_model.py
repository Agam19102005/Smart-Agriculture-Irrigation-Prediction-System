"""Optional: register the locally trained model in an Azure ML workspace."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "azure_config.json"
MODEL_PATH = ROOT / "models" / "irrigation_model.joblib"

def main():
    if not CONFIG_PATH.exists():
        print("Missing azure_config.json. Copy azure_config.example.json to azure_config.json and fill in your workspace details.")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print("Model not found. Run `python train_model.py` first.")
        sys.exit(1)

    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("Azure SDK missing. Run `pip install -r requirements-azure.txt`.")
        sys.exit(1)

    cfg = json.loads(CONFIG_PATH.read_text())
    required = ["subscription_id", "resource_group", "workspace_name"]
    missing = [key for key in required if not cfg.get(key) or str(cfg[key]).startswith("YOUR_")]
    if missing:
        print("Fill these fields in azure_config.json: " + ", ".join(missing))
        sys.exit(1)

    ml_client = MLClient(
        DefaultAzureCredential(),
        cfg["subscription_id"],
        cfg["resource_group"],
        cfg["workspace_name"],
    )
    model = Model(
        path=str(MODEL_PATH),
        name="smart-agriculture-irrigation-model",
        description="Random Forest model for simulated smart agriculture irrigation prediction",
        type="custom_model",
        tags={"project": "AIEC-mini-project", "inference": "local-edge-simulation"},
    )
    registered = ml_client.models.create_or_update(model)
    print(f"Registered Azure ML model: {registered.name}")
    print(f"Version: {registered.version}")
    print("Note: registration manages the model in Azure; local dashboard inference remains hardware-free.")

if __name__ == "__main__":
    main()

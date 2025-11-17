#this script retroactively logs saved models and their metrics to MLflow
import os
import json
import numpy as np
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import load_model

# Path to your models directory
MODELS_BASE = "./models"

# Set MLflow experiment
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Malaria_Classification")

for model_name in os.listdir(MODELS_BASE):
    model_dir = os.path.join(MODELS_BASE, model_name)
    if not os.path.isdir(model_dir):
        continue

    model_path = os.path.join(model_dir, "model.keras")
    history_path = os.path.join(model_dir, "logs", "history.json")
    report_path = os.path.join(model_dir, "reports", "classification_report.json")
    conf_matrix_path = os.path.join(model_dir, "reports", "confusion_matrix.npy")

    if not os.path.exists(model_path):
        continue  # skip models without saved file

    print(f"Logging model: {model_name}")

    with mlflow.start_run(run_name=model_name):
        # Log parameters inferred from folder name
        tokens = model_name.split("_")

        # Initialize defaults
        input_type = None
        augmentation = None
        dual_input = False
        layers = None
        lr = None
        es_pat = None
        rlrop_pat = None

        for i, t in enumerate(tokens):
            # Input types
            if t in ["rgb", "rgbh", "rgbs", "hsv"]:
                input_type = t
            elif t == "clahe":
                augmentation = "clahe"

            # Dual-input
            elif t == "dual":
                dual_input = True

            # Layers
            elif t.endswith("l") and t[:-1].isdigit():
                layers = int(t[:-1])

            # Learning rate
            elif t == "lr" and i + 1 < len(tokens):
                try:
                    lr = float(tokens[i + 1])
                except ValueError:
                    pass

            # EarlyStopping patience
            elif t.startswith("es") and "pat" in t:
                es_pat = int(t.split("pat")[-1])

            # ReduceLROnPlateau patience
            elif t.startswith("rlrop") and "pat" in t:
                rlrop_pat = int(t.split("pat")[-1])

        # --- Log parameters safely ---
        if input_type:
            mlflow.log_param("input_type", input_type)
        if augmentation:
            mlflow.log_param("augmentation", augmentation)
        mlflow.log_param("dual_input", dual_input)
        if layers:
            mlflow.log_param("layers", layers)
        if lr:
            mlflow.log_param("learning_rate", lr)
        if es_pat:
            mlflow.log_param("es_patience", es_pat)
        if rlrop_pat:
            mlflow.log_param("rlrop_patience", rlrop_pat)

        # 🟩 --- Paste the new block here ---
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)

            # Handle per-class (blank key) if present
            if "" in report:
                cls = report[""]
                mlflow.log_metric("precision_class", cls.get("precision", 0.0))
                mlflow.log_metric("recall_class", cls.get("recall", 0.0))
                mlflow.log_metric("f1_class", cls.get("f1-score", 0.0))

            # Handle averages
            if "macro avg" in report:
                mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
                mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
                mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

            if "weighted avg" in report:
                mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
                mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])
                mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])

            # Handle accuracy (dict format)
            if "accuracy" in report:
                acc = report["accuracy"]
                if isinstance(acc, dict):
                    mlflow.log_metric("accuracy", acc.get("accuracy", 0.0))
                else:
                    mlflow.log_metric("accuracy", acc)
        # 🟩 --- End of new block ---

        # Log confusion matrix
        if os.path.exists(conf_matrix_path):
            mlflow.log_artifact(conf_matrix_path, artifact_path="confusion_matrix")

        
            cm = np.load(conf_matrix_path)
            if cm.shape == (2, 2):  

                # Extract counts
                tn, fp, fn, tp = cm.ravel()

                # Log metrics to MLflow
                mlflow.log_metric("true_negatives", int(tn))
                mlflow.log_metric("false_positives", int(fp))
                mlflow.log_metric("false_negatives", int(fn))
                mlflow.log_metric("true_positives", int(tp))


        # Log training history
        if os.path.exists(history_path):
            mlflow.log_artifact(history_path, artifact_path="training_history")

        # Log model itself
        model = load_model(model_path)
        mlflow.tensorflow.log_model(model, artifact_path="model")

print("✅ Finished retroactive logging to MLflow.")
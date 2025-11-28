import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os
import io
import traceback

MODEL_FILENAME = "parkinson_tremor_model .pth"
TARGET_NAME = "Rest_tremor"

# Load model
model = None
model_load_error = None
if os.path.exists(MODEL_FILENAME):
    try:
        model = joblib.load(MODEL_FILENAME)
    except Exception as e:
        model_load_error = f"Failed to load model: {e}\n{traceback.format_exc()}"
else:
    model_load_error = f"Model file not found: {MODEL_FILENAME}"

# Use the model's feature names (from training) if possible
MODEL_FEATURES = None
if model is not None:
    try:
        MODEL_FEATURES = model.feature_names_in_.tolist()
    except Exception:
        MODEL_FEATURES = None

LABEL_MAP = {
    0: "No Tremor",
    1: "Tremor (mild)",
    2: "Tremor (moderate)",
    3: "Tremor (severe)"
}

def _safe_read_csv(uploaded_file):
    try:
        if hasattr(uploaded_file, "name"):
            return pd.read_csv(uploaded_file.name)
        else:
            content = uploaded_file.read()
            return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Unable to read CSV: {e}")

def predict_from_csv(file) -> (pd.DataFrame, str):
    if model is None:
        return None, f"ERROR: Model not available. {model_load_error}"

    try:
        df = _safe_read_csv(file)

        # Drop target if present
        if TARGET_NAME in df.columns:
            df = df.drop(columns=[TARGET_NAME])

        # Ensure columns match model's feature names
        if MODEL_FEATURES:
            missing = [f for f in MODEL_FEATURES if f not in df.columns]
            if missing:
                raise ValueError(f"Missing features in CSV: {missing}")
            df = df[MODEL_FEATURES]  # reorder columns

        preds = model.predict(df)
        try:
            proba_all = model.predict_proba(df)
            probs = []
            for i, p in enumerate(preds):
                classes = model.classes_
                idx = list(classes).index(p) if p in classes else np.argmax(proba_all[i])
                probs.append(proba_all[i][idx])
            probs = np.array(probs)
        except Exception:
            probs = np.array([np.nan]*len(preds))

        out_df = df.copy()
        out_df["model_prediction_raw"] = preds
        out_df["prediction_confidence"] = probs
        out_df["model_prediction_label"] = out_df["model_prediction_raw"].apply(
            lambda r: LABEL_MAP.get(int(r), f"Label {r}") if pd.notna(r) else str(r)
        )

        status = f"Predicted {len(out_df)} rows using {df.shape[1]} features."
        return out_df, status

    except Exception as e:
        tb = traceback.format_exc()
        return None, f"ERROR processing file: {e}\n{tb}"

def manual_prediction(**kwargs):
    """
    Manual input must include all features the model expects.
    kwargs keys should match MODEL_FEATURES.
    """
    if model is None:
        return "ERROR: Model not available. " + str(model_load_error), float("nan")

    try:
        if MODEL_FEATURES is None:
            return "ERROR: Model feature names unknown. Cannot use manual input.", float("nan")

        # Build single-row DataFrame
        row = []
        for f in MODEL_FEATURES:
            if f in kwargs:
                row.append(float(kwargs[f]))
            else:
                return f"ERROR: Missing manual input for feature: {f}", float("nan")

        df = pd.DataFrame([row], columns=MODEL_FEATURES)
        pred = model.predict(df)[0]

        try:
            proba_all = model.predict_proba(df)[0]
            classes = model.classes_
            idx = list(classes).index(pred) if pred in classes else np.argmax(proba_all)
            prob = float(proba_all[idx])
        except Exception:
            prob = float("nan")

        label_text = LABEL_MAP.get(int(pred), f"Label {pred}")
        return label_text, prob
    except Exception as e:
        tb = traceback.format_exc()
        return f"ERROR in manual prediction: {e}\n{tb}", float("nan")

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("# Neuro Tracker — Tremor Detection")
    if model is None:
        gr.Markdown(f"**Model load error:** `{model_load_error}`")
    else:
        gr.Markdown(f"**Model loaded successfully. Expecting {len(MODEL_FEATURES)} features for prediction.**")

    with gr.Tab("CSV Prediction"):
        csv_file = gr.File(label="Upload full feature CSV")
        csv_out = gr.Dataframe(label="Predictions")
        csv_status = gr.Textbox(label="Status / Errors", interactive=False)
        csv_btn = gr.Button("Run CSV Prediction")

        def on_csv_submit(f):
            out, status = predict_from_csv(f)
            if out is None:
                return pd.DataFrame(), status
            # move label/confidence to end
            cols = [c for c in out.columns if c not in ("model_prediction_label","prediction_confidence")]
            final_order = cols + ["model_prediction_label","prediction_confidence"]
            return out[final_order].head(200), status

        csv_btn.click(on_csv_submit, inputs=csv_file, outputs=[csv_out, csv_status])

    if MODEL_FEATURES is not None:
        with gr.Tab("Manual Input"):
            # create a Number input for each model feature
            inputs = []
            for f in MODEL_FEATURES:
                inputs.append(gr.Number(label=f, value=0.0))
            man_btn = gr.Button("Predict")
            man_label = gr.Textbox(label="Prediction (label)")
            man_conf = gr.Number(label="Prediction confidence (0-1)")
            man_btn.click(
                lambda *args: manual_prediction(**dict(zip(MODEL_FEATURES, args))),
                inputs=inputs,
                outputs=[man_label, man_conf]
            )

if __name__ == "__main__":
    demo.launch()

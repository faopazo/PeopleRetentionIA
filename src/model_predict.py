import os
import joblib
import pandas as pd

def load_artifacts(model_dir: str):
    model = joblib.load(os.path.join(model_dir, "attrition_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return model, scaler

def predict(df: pd.DataFrame, model_dir: str):
    model, scaler = load_artifacts(model_dir)
    # Si el DataFrame tiene la columna target, quitarla
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
    else:
        X = df.copy()
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = model.predict(X_scaled)
    return pd.DataFrame({'prediction': pred, 'probability': proba}, index=df.index)

"""src/model_predict.py
Carga artefactos y predice sobre un DataFrame o CSV.
"""
import os
import joblib
import pandas as pd

def load_artifacts(model_dir: str):
    model = joblib.load(os.path.join(model_dir, 'attrition_model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    return model, scaler

def predict(df: pd.DataFrame, model_dir: str):
    model, scaler = load_artifacts(model_dir)
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
    else:
        X = df.copy()
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:,1]
    pred = model.predict(X_scaled)
    return pd.DataFrame({'prediction': pred, 'probability': proba}, index=df.index)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV processed input')
    parser.add_argument('--model-dir', default='models', help='Model directory')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    res = predict(df, args.model_dir)
    res.to_csv(args.output, index=False)
    print(f'Predictions written to {args.output}')

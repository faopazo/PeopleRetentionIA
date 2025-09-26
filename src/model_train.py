"""src/model_train.py
Entrena un clasificador RandomForest sobre data/processed.csv y guarda modelo y scaler.
"""
import argparse
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

def main(input_csv: str, model_outdir: str):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Processed file not found: {input_csv}")
    df = pd.read_csv(input_csv)
    if 'target' not in df.columns:
        raise ValueError("La columna 'target' no está presente en el dataset procesado")
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:,1]
    print('Classification report:')
    print(classification_report(y_test, y_pred))
    print('ROC AUC:', roc_auc_score(y_test, y_proba))
    os.makedirs(model_outdir, exist_ok=True)
    joblib.dump(model, os.path.join(model_outdir, 'attrition_model.joblib'))
    joblib.dump(scaler, os.path.join(model_outdir, 'scaler.joblib'))
    print(f'Model and scaler saved to {model_outdir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed.csv')
    parser.add_argument('--model-output', default='models')
    args = parser.parse_args()
    main(args.input, args.model_output)

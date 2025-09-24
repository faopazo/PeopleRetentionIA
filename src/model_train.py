import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_pipeline(input_csv: str, model_dir: str):
    df = pd.read_csv(input_csv)
    if 'target' not in df.columns:
        raise ValueError("La columna 'target' no está presente en el dataset procesado")
    X = df.drop(columns=['target'])
    y = df['target']
    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Modelo
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]
    print("=== Reporte clasificación ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    # Guardar artefactos
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "attrition_model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    print(f"Modelo y scaler guardados en {model_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="CSV procesado input")
    parser.add_argument('--model-output', required=True, help="Directorio donde guardar modelo y scaler")
    args = parser.parse_args()
    train_pipeline(args.input, args.model_output)

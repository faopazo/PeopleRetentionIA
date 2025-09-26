"""src/data_prep.py
Limpieza y feature engineering para el dataset HR Employee Attrition.
Por defecto lee data/HR_Employee_Attrition.csv y escribe data/processed.csv
"""
import argparse
import os
import pandas as pd
import numpy as np

CATEGORICAL = [
    'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'
]
NUMERIC = [
    'Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','MonthlyIncome',
    'NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears','TrainingTimesLastYear',
    'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'
]

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Attrition' in df.columns:
        df['target'] = df['Attrition'].map({'Yes':1, 'No':0})
    if 'TotalWorkingYears' in df.columns and 'YearsAtCompany' in df.columns:
        df['tenure_ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'].replace(0, np.nan))
        df['tenure_ratio'] = df['tenure_ratio'].fillna(0)
    cats_present = [c for c in CATEGORICAL if c in df.columns]
    df = pd.get_dummies(df, columns=cats_present, drop_first=True)
    keep = [c for c in df.columns if c in NUMERIC or c.startswith(tuple([c_ + '_' for c_ in cats_present])) or c in ['tenure_ratio','target']]
    if 'target' in df.columns:
        cols = [c for c in df.columns if (c in keep or c=='target')]
    else:
        cols = [c for c in df.columns if c in (NUMERIC + ['tenure_ratio'])]
    return df[cols]

def main(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Cargando datos: {df.shape[0]} filas, {df.shape[1]} columnas")
    processed = feature_engineering(basic_clean(df))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed.to_csv(output_path, index=False)
    print(f"Datos procesados guardados en {output_path}, shape={processed.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/HR_Employee_Attrition.csv')
    parser.add_argument('--output', default='data/processed.csv')
    args = parser.parse_args()
    main(args.input, args.output)

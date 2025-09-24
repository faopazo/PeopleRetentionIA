import argparse
import pandas as pd
import numpy as np

# Define columnas categóricas / numéricas esperadas
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
            df[col] = df[col].fillna("Unknown")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Convertir la variable objetivo
    if 'Attrition' in df.columns:
        df['target'] = df['Attrition'].map({'Yes':1, 'No':0})
    # Generar alguna característica compuesta
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        # evitar división por cero
        df['tenure_ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'].replace(0, np.nan))
        df['tenure_ratio'] = df['tenure_ratio'].fillna(0.0)
    # One-hot encoding
    cats_present = [c for c in CATEGORICAL if c in df.columns]
    df = pd.get_dummies(df, columns=cats_present, drop_first=True)
    # Escoger columnas finales
    # mantener todas las numéricas + las dummies + target
    cols_keep = []
    for c in df.columns:
        if c in NUMERIC or c.startswith(tuple([c_ + "_" for c_ in cats_present])) or c in ['target','tenure_ratio']:
            cols_keep.append(c)
    # Asegurar que target esté al final
    if 'target' in cols_keep:
        cols_keep = [c for c in cols_keep if c != 'target'] + ['target']
    return df[cols_keep]

def main(args):
    df = pd.read_csv(args.input)
    df = basic_clean(df)
    processed = feature_engineering(df)
    processed.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}, shape: {processed.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="CSV raw input file")
    parser.add_argument('--output', required=True, help="CSV processed output file")
    args = parser.parse_args()
    main(args)

# PeopleRetentionIA

Proyecto completo para People Analytics: predicción de attrition usando el dataset de Kaggle.

## Estructura
```
PeopleRetentionIA/
├── data/                      # datos crudos y procesados
├── src/                       # scripts: preproc, train, predict
├── app/                       # Streamlit app
├── models/                    # modelos entrenados (están en .gitignore)
├── notebooks/                 # notebooks de EDA
├── requirements.txt
└── README.md
```

## Cómo reproducir (local)

1. Crear entorno e instalar deps:
```bash
python -m venv .venv
source .venv/bin/activate   # o .\.venv\Scripts\activate en Windows PowerShell
pip install -r requirements.txt
```

2. Preprocesar (archivo `data/HR_Employee_Attrition.csv`):
```bash
python src/data_prep.py --input data/HR_Employee_Attrition.csv --output data/processed.csv
```

3. Entrenar modelo:
```bash
python src/model_train.py --input data/processed.csv --model-output models/
```

4. Ejecutar app Streamlit:
```bash
streamlit run app/app_streamlit.py
```


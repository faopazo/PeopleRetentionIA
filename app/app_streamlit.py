"""app/app_streamlit.py
Streamlit app to explore processed data and show predictions (uses models/attrition_model.joblib).
"""
import streamlit as st
import pandas as pd
import os
from src.model_predict import predict

st.set_page_config(page_title='PeopleRetentionIA', layout='wide')
st.title('PeopleRetentionIA — Risk of Attrition')

MODEL_DIR = 'models'

st.sidebar.header('Cargar datos (processed)')
uploaded = st.sidebar.file_uploader('Sube un CSV preprocesado (processed.csv)', type=['csv'])

if uploaded is None:
    default = 'data/processed.csv'
    if os.path.exists(default):
        df = pd.read_csv(default)
        st.sidebar.write('Usando data/processed.csv')
    else:
        st.error('No se encontró data/processed.csv. Corre data_prep.py primero o subí un CSV.')
        st.stop()
else:
    df = pd.read_csv(uploaded)

if st.checkbox('Mostrar datos (first 200 rows)'):
    st.dataframe(df.head(200))

st.header('Distribuciones')
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c!='target']
if num_cols:
    sel = st.selectbox('Variable numérica', options=num_cols)
    fig = None
    try:
        import plotly.express as px
        fig = px.histogram(df, x=sel, color=('target' if 'target' in df.columns else None), marginal='box')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.write('No se pudo generar gráfico interactivo:', e)

st.header('Predicciones de riesgo')
if not os.path.exists(MODEL_DIR):
    st.warning('No se encontró la carpeta models/. Entrena el modelo con src/model_train.py')
else:
    if st.button('Predecir para todo el dataset'):
        preds = predict(df, MODEL_DIR)
        res = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        st.dataframe(res.sort_values('probability', ascending=False).head(50))
        st.download_button('Descargar predicciones', data=res.to_csv(index=False), file_name='predictions.csv')



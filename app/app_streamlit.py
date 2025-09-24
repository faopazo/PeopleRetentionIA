import streamlit as st
import pandas as pd
import plotly.express as px
from src.model_predict import predict, load_artifacts
import os

MODEL_DIR = "models"

st.set_page_config(page_title="PeopleRetentionIA", layout="wide")

st.title("PeopleRetentionIA — Risk of Attrition")

st.sidebar.header("Carga de datos")
uploaded = st.sidebar.file_uploader("Sube un CSV procesado (processed.csv)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # intenta cargar data/processed.csv
    default_path = "data/processed.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.sidebar.write("Cargando `data/processed.csv`")
    else:
        st.error("No se encontró un archivo procesado. Sube tu CSV o genera `data/processed.csv`.")
        st.stop()

if 'target' in df.columns:
    show_target = True
else:
    show_target = False

# Mostrar tabla
if st.checkbox("Mostrar datos"):
    st.dataframe(df.head(100))

# Distribuciones
st.header("Exploración de variables")
col1, col2 = st.columns(2)
with col1:
    # seleccionar variable numérica
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'target']
    sel = st.selectbox("Variable numérica", numeric_cols)
    fig = px.histogram(df, x=sel, color=("target" if show_target else None), marginal="box")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if show_target:
        fig2 = px.pie(df, names=df["target"].map({1: "Attrition", 0: "No Attrition"}))
        st.plotly_chart(fig2, use_container_width=True)

# Predicción
st.header("Predicciones de riesgo")
if os.path.exists(MODEL_DIR):
    if st.button("Predecir para todas las filas"):
        preds = predict(df, MODEL_DIR)
        result = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        st.dataframe(result.sort_values("probability", ascending=False).head(20))
        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar predicciones", data=csv, file_name="predictions.csv", mime="text/csv")
else:
    st.warning("No se encontró el modelo entrenado en carpeta `models/`.")

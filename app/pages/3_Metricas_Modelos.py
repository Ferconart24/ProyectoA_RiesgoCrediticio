import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src import config

st.set_page_config(page_title="Métricas de Modelos", page_icon="📊")
st.title("📊 Métricas de Rendimiento de los Modelos")

models_dir = config.MODELS_DIR

st.subheader("Modelo Binario")

col1, col2 = st.columns(2)

with col1:
    if (models_dir / "binary_confusion_matrix.png").exists():
        st.image(str(models_dir / "binary_confusion_matrix.png"), caption="Matriz de Confusión")
    else:
        st.info("No se encontró la matriz de confusión binaria")

with col2:
    if (models_dir / "binary_roc_curve.png").exists():
        st.image(str(models_dir / "binary_roc_curve.png"), caption="Curva ROC")
    else:
        st.info("No se encontró la curva ROC")

# Si guardaste el history en CSV
if (models_dir / "binary_history.csv").exists():
    import pandas as pd
    history = pd.read_csv(models_dir / "binary_history.csv")
    st.line_chart(history[['accuracy', 'val_accuracy']])

st.subheader("Modelo Multiclase")

if (models_dir / "multiclass_confusion_matrix.png").exists():
    st.image(str(models_dir / "multiclass_confusion_matrix.png"), caption="Matriz de Confusión Multiclase")
else:
    st.info("No se encontró la matriz de confusión multiclase")

# Mostrar comparación si existe
if (models_dir / "comparison.csv").exists():
    comp = pd.read_csv(models_dir / "comparison.csv")
    st.subheader("Comparación de Modelos")
    st.dataframe(comp)
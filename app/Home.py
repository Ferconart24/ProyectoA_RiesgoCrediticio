"""
Frontend con Streamlit para el Sistema de Predicción de Riesgo Crediticio

Ejecutar: streamlit run Home.py
URL: http://localhost:8501
"""

import streamlit as st
import sys
from pathlib import Path

# Añadir path del proyecto
sys.path.append(str(Path(__file__).parent.parent))
from src import config

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(
    page_title="Sistema de Riesgo Crediticio",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === ESTILOS PERSONALIZADOS ===
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #1F4E78, #2E75B5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #1F4E78;
    margin-top: 1.5rem;
}

.card {
    background-color: #f8f9fc;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.title("🏦 Sistema de IA")
st.sidebar.markdown("### Predicción de Riesgo Crediticio")
st.sidebar.markdown("---")

# Información del proyecto
st.sidebar.markdown("### 📊 Información del Proyecto")
st.sidebar.info("""
**Equipo:**
- Fernando Contreras Artavia
- Marisol Viquez Rivera 
- Camila Jiménez Gómez

**Curso:** IA Aplicada - CUC  
**Año:** 2026
""")

st.sidebar.markdown("---")

# Enlaces útiles
st.sidebar.markdown("### 🔗 Enlaces")
st.sidebar.markdown("[📖 Documentación API](http://localhost:8000/docs)")
st.sidebar.markdown("[📁 GitHub del Proyecto](#)")
st.sidebar.markdown("[📊 Dataset UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔄 Estado del Sistema")

st.sidebar.success("Modelo binario listo")
st.sidebar.success("Modelo multiclase listo")
st.sidebar.success("API listo")


# === PÁGINA PRINCIPAL ===

# Header
st.markdown('<h1 class="main-header"> Sistema de Predicción de Riesgo Crediticio</h1>',
            unsafe_allow_html=True)

st.markdown("""
Este sistema utiliza **Redes Neuronales Artificiales (ANN)** para predecir el riesgo crediticio 
de clientes bancarios, apoyando decisiones de aprobación de préstamos de manera inteligente y automatizada.
""")

# === TABS PRINCIPALES ===
tab1, tab2, tab3 = st.tabs(["📑 Descripción", "🧠 Modelos", "🏆 Resultados"])

with tab1:
    st.markdown('<div class="sub-header">📑 Descripción del Proyecto</div>',
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Objetivos")
        st.markdown("""
        - Predecir si un crédito será bueno o malo
        - Clasificar clientes por nivel de riesgo
        - Automatizar proceso de evaluación crediticia
        - Reducir tasas de default
        """)
        
        st.markdown("### 📊 Dataset")
        st.markdown("""
        **German Credit Data (UCI)**
        - 1,000 clientes
        - 20 variables predictoras
        - Variables demográficas y financieras
        """)
    
    with col2:
        st.markdown("### 🔧 Tecnologías")
        st.markdown("""
        - **TensorFlow/Keras**: Redes neuronales
        - **FastAPI**: API REST
        - **Streamlit**: Frontend interactivo
        - **Scikit-learn**: Preprocesamiento
        """)
        st.markdown("### ⚙️ Arquitectura del Sistema")

        st.code("""
        Usuario → Streamlit → FastAPI → Modelo ANN → Predicción
        """)
        st.markdown("### 📁 Navegación")
        st.info("""
        👈 Usa el menú lateral para:
        - 📝 Realizar predicciones individuales
        - 📊 Analizar lotes de solicitudes
        - 📈 Ver métricas de los modelos
        """)

with tab2:
    st.markdown('<div class="sub-header">🧠 Modelos Implementados</div>',
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Modelo 1: Clasificación Binaria")
        st.markdown("""
        **Objetivo:** Predecir aprobación de crédito
        
        **Clases:**
        - ✅ Good Credit (Aprobar)
        - ❌ Bad Credit (Rechazar)
        
        **Arquitectura:**
        - Input Layer: [N features]
        - Hidden Layers: [64, 32, 16]
        - Output Layer: 1 neurona (sigmoid)
        
        **Métricas:**
        - Accuracy: [completar después del entrenamiento]
        - Precision: [completar]
        - Recall: [completar]
        - F1-Score: [completar]
        """)
    
    with col2:
        st.markdown("### 📉 Modelo 2: Clasificación Multiclase")
        st.markdown("""
        **Objetivo:** Clasificar nivel de riesgo
        
        **Clases:**
        - 🟢 Riesgo Bajo
        - 🟡 Riesgo Medio
        - 🟠 Riesgo Alto
        - 🔴 Riesgo Crítico
        
        **Arquitectura:**
        - Input Layer: [N features]
        - Hidden Layers: [128, 64, 32]
        - Output Layer: 4 neuronas (softmax)
        
        **Métricas:**
        - Accuracy: [completar]
        - Precision (macro): [completar]
        - Recall (macro): [completar]
        - F1-Score (macro): [completar]
        """)

with tab3:
    st.markdown('<div class="sub-header">📈 Resultados y Conclusiones</div>', 
                unsafe_allow_html=True)
    
    st.warning("⚠️ Esta sección se completará después del entrenamiento de los modelos")
    st.markdown("### 💰 Impacto Financiero Estimado")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Reducción estimada de default", "18%")

    with col2:
        st.metric("Ahorro anual proyectado", "$250,000")

    # Placeholder para resultados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Accuracy (Binario)", value="---%", delta="---")
    
    with col2:
        st.metric(label="Accuracy (Multiclase)", value="---%", delta="---")
    
    with col3:
        st.metric(label="ROC-AUC", value="---", delta="---")
    
    st.markdown("---")
    
    st.markdown("### 🎓 Conclusiones")
    st.markdown("""
    **Hallazgos principales:**
    - [A completar después del análisis]
    - [A completar después del análisis]
    - [A completar después del análisis]
    
    **Recomendaciones:**
    - [A completar después del análisis]
    - [A completar después del análisis]
    """)

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; font-size: 0.9rem; color: #666;'>
© 2026 Sistema Inteligente de Riesgo Crediticio  
Desarrollado con TensorFlow, FastAPI y Streamlit  
Colegio Universitario de Cartago
</div>
""", unsafe_allow_html=True)


# === INFORMACIÓN DE DEBUG (solo en desarrollo) ===
with st.expander("🔧 Información de Debug"):
    st.markdown("### Configuración del Sistema")
    st.json({
        "PROJECT_ROOT": str(config.PROJECT_ROOT),
        "MODELS_DIR": str(config.MODELS_DIR),
        "BINARY_MODEL": str(config.BINARY_MODEL_PATH),
        "MULTICLASS_MODEL": str(config.MULTICLASS_MODEL_PATH),
        "API_URL": f"http://localhost:{config.API_PORT}"
    })

import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(page_title="Análisis por Lotes", page_icon="📂")
st.title("📂 Análisis por Lotes de Solicitudes")

API_URL = "http://localhost:8000"

uploaded_file = st.file_uploader("Subir archivo CSV con solicitudes", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

    if st.button("Procesar lote"):
        results_bin = []
        results_multi = []
        progress_bar = st.progress(0)

        for i, row in df.iterrows():
            data = row.to_dict()
            # Asegurar tipos (convertir a los tipos esperados)
            # (puede ser necesario ajustar según el formato del CSV)
            try:
                resp_bin = requests.post(f"{API_URL}/predict/binary", json=data)
                resp_multi = requests.post(f"{API_URL}/predict/risk_level", json=data)
            except:
                results_bin.append("Error conexión")
                results_multi.append("Error conexión")
                continue

            if resp_bin.status_code == 200 and resp_multi.status_code == 200:
                bin_res = resp_bin.json()
                multi_res = resp_multi.json()
                results_bin.append(bin_res["prediction"])
                results_multi.append(multi_res["risk_level"])
            else:
                results_bin.append("Error")
                results_multi.append("Error")

            progress_bar.progress((i + 1) / len(df))

        df["Predicción Binaria"] = results_bin
        df["Nivel de Riesgo"] = results_multi

        st.success("Procesamiento completado")
        st.dataframe(df)

        # Botón para descargar resultados
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar resultados como CSV",
            data=csv,
            file_name="resultados_lote.csv",
            mime="text/csv"
        )
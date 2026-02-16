import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Predicción Individual", page_icon="🔮")
st.title("🔮 Predicción Individual de Riesgo Crediticio")

# URL de la API (ajusta si es necesario)
API_URL = "http://localhost:8000"

st.markdown("Complete los datos del solicitante:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        checking_status = st.selectbox("Estado de cuenta corriente",
                                        ["A11", "A12", "A13", "A14"])
        duration = st.number_input("Duración (meses)", min_value=1, max_value=72, value=12)
        credit_history = st.selectbox("Historial crediticio",
                                        ["A30", "A31", "A32", "A33", "A34"])
        purpose = st.selectbox("Propósito del crédito",
                                ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"])
        credit_amount = st.number_input("Monto del crédito", min_value=100, value=1000)
        savings_status = st.selectbox("Estado de ahorros",
                                        ["A61", "A62", "A63", "A64", "A65"])
        employment = st.selectbox("Situación laboral",
                                    ["A71", "A72", "A73", "A74", "A75"])
        installment_rate = st.number_input("Tasa de cuota (%)", min_value=1, max_value=100, value=4)
        personal_status = st.selectbox("Estado civil / género",
                                        ["A91", "A92", "A93", "A94", "A95"])
        other_parties = st.selectbox("Otros deudores", ["A101", "A102", "A103"])

    with col2:
        residence_since = st.number_input("Residencia desde (años)", min_value=0, max_value=100, value=2)
        property_magnitude = st.selectbox("Propiedad", ["A121", "A122", "A123", "A124"])
        age = st.number_input("Edad", min_value=18, max_value=100, value=35)
        other_payment_plans = st.selectbox("Otros planes de pago", ["A141", "A142", "A143"])
        housing = st.selectbox("Vivienda", ["A151", "A152", "A153"])
        existing_credits = st.number_input("Créditos existentes", min_value=0, max_value=10, value=1)
        job = st.selectbox("Tipo de trabajo", ["A171", "A172", "A173", "A174"])
        num_dependents = st.number_input("Número de dependientes", min_value=0, max_value=10, value=0)
        own_telephone = st.selectbox("Teléfono propio", ["A191", "A192"])
        foreign_worker = st.selectbox("Trabajador extranjero", ["A201", "A202"])

    submitted = st.form_submit_button("Predecir")

if submitted:
    # Construir el payload
    data = {
        "checking_status": checking_status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_status": savings_status,
        "employment": employment,
        "installment_rate": installment_rate,
        "personal_status": personal_status,
        "other_parties": other_parties,
        "residence_since": residence_since,
        "property_magnitude": property_magnitude,
        "age": age,
        "other_payment_plans": other_payment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "num_dependents": num_dependents,
        "own_telephone": own_telephone,
        "foreign_worker": foreign_worker
    }

    try:
        # Llamar a la API para predicción binaria
        response_bin = requests.post(f"{API_URL}/predict/binary", json=data)
        response_multi = requests.post(f"{API_URL}/predict/risk_level", json=data)

        if response_bin.status_code == 200 and response_multi.status_code == 200:
            bin_result = response_bin.json()
            multi_result = response_multi.json()

            st.success("Predicción realizada con éxito")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Clasificación Binaria")
                st.metric("Predicción", bin_result["prediction"])
                st.metric("Probabilidad Bad", f"{bin_result['probability_bad']:.2%}")
                st.metric("Probabilidad Good", f"{bin_result['probability_good']:.2%}")
                st.metric("Confianza", f"{bin_result['confidence']:.2%}")

            with col2:
                st.subheader("📈 Nivel de Riesgo")
                st.metric("Nivel", multi_result["risk_level"])
                st.write("**Probabilidades:**")
                for k, v in multi_result["probabilities"].items():
                    st.write(f"- {k}: {v:.2%}")
                st.info(f"**Recomendación**: {multi_result['recommendation']}")
        else:
            st.error("Error al conectar con la API. Verifica que esté corriendo.")
    except Exception as e:
        st.error(f"Error de conexión: {e}")
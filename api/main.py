"""
API REST con FastAPI para el Sistema de Predicción de Riesgo Crediticio

Endpoints:
- GET /: Información de la API
- GET /ping: Verificación básica de conectividad
- POST /predict/binary: Predicción binaria (Good/Bad credit)
- POST /predict/risk_level: Predicción de nivel de riesgo
- GET /health: Estado del servidor

Ejecutar: uvicorn main:app --reload
Documentación: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import joblib
from tensorflow import keras
import traceback

# Añadir path del proyecto
sys.path.append(str(Path(__file__).parent.parent))
from src import config

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Predicción de Riesgo Crediticio",
    description="API para predicción de riesgo crediticio usando ANN",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MODELOS DE DATOS (SCHEMAS) ===

class CreditApplication(BaseModel):
    """Schema para solicitud de crédito"""
    checking_status: str = Field(..., description="Estado de cuenta corriente")
    duration: int = Field(..., description="Duración del crédito en meses")
    credit_history: str = Field(..., description="Historial crediticio")
    purpose: str = Field(..., description="Propósito del crédito")
    credit_amount: float = Field(..., description="Monto del crédito")
    savings_status: str = Field(..., description="Estado de ahorros")
    employment: str = Field(..., description="Situación laboral")
    installment_rate: int = Field(..., description="Tasa de cuota")
    personal_status: str = Field(..., description="Estado civil y género")
    other_parties: str = Field(..., description="Otros deudores")
    residence_since: int = Field(..., description="Residencia desde")
    property_magnitude: str = Field(..., description="Propiedad")
    age: int = Field(..., description="Edad")
    other_payment_plans: str = Field(..., description="Otros planes de pago")
    housing: str = Field(..., description="Vivienda")
    existing_credits: int = Field(..., description="Créditos existentes")
    job: str = Field(..., description="Tipo de trabajo")
    num_dependents: int = Field(..., description="Número de dependientes")
    own_telephone: str = Field(..., description="Teléfono propio")
    foreign_worker: str = Field(..., description="Trabajador extranjero")

    class Config:
        json_schema_extra = {
            "example": {
                "checking_status": "A11",
                "duration": 6,
                "credit_history": "A34",
                "purpose": "A43",
                "credit_amount": 1169,
                "savings_status": "A65",
                "employment": "A75",
                "installment_rate": 4,
                "personal_status": "A93",
                "other_parties": "A101",
                "residence_since": 4,
                "property_magnitude": "A121",
                "age": 67,
                "other_payment_plans": "A143",
                "housing": "A152",
                "existing_credits": 2,
                "job": "A173",
                "num_dependents": 1,
                "own_telephone": "A192",
                "foreign_worker": "A201"
            }
        }

class BinaryPredictionResponse(BaseModel):
    """Respuesta de predicción binaria"""
    prediction: str = Field(..., description="Good o Bad")
    probability_bad: float = Field(..., description="Probabilidad de mal crédito")
    probability_good: float = Field(..., description="Probabilidad de buen crédito")
    confidence: float = Field(..., description="Confianza de la predicción")

class RiskLevelResponse(BaseModel):
    """Respuesta de nivel de riesgo"""
    risk_level: str = Field(..., description="Bajo, Medio, Alto, Crítico")
    probabilities: dict = Field(..., description="Probabilidades por nivel")
    recommendation: str = Field(..., description="Recomendación")

# === CARGAR MODELOS Y PREPROCESADORES ===
binary_model = None
multiclass_model = None
scaler = None
encoders = None

try:
    binary_model = keras.models.load_model(config.BINARY_MODEL_PATH)
    multiclass_model = keras.models.load_model(config.MULTICLASS_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    encoders = joblib.load(config.LABEL_ENCODER_PATH)
    print("✓ Modelos y preprocesadores cargados correctamente")
    if binary_model is not None:
        print(f"  Modelo binario espera {binary_model.input_shape[1]} características")
    if multiclass_model is not None:
        print(f"  Modelo multiclase espera {multiclass_model.input_shape[1]} características")
except Exception as e:
    print(f"⚠️ Error cargando modelos: {e}")
    print("Los endpoints de predicción no funcionarán hasta resolver el error")

# === FUNCIONES AUXILIARES ===

def preprocess_input(data: CreditApplication) -> np.ndarray:
    """
    Preprocesa los datos de entrada:
    - Codifica las variables categóricas con los LabelEncoders guardados.
    - Escala las variables numéricas con el StandardScaler.
    - Concatena en el orden: numéricas primero, luego categóricas.
    Retorna un array numpy listo para el modelo.
    """
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # 1. Codificar variables categóricas
    cat_values = []
    for col in config.CATEGORICAL_COLUMNS:
        if col not in encoders:
            raise ValueError(f"No se encontró encoder para la columna '{col}'")
        value = df[col].iloc[0]
        if value not in encoders[col].classes_:
            raise ValueError(
                f"Valor '{value}' no reconocido para la columna '{col}'. "
                f"Valores permitidos: {list(encoders[col].classes_)}"
            )
        encoded = encoders[col].transform([value])[0]
        cat_values.append(encoded)
    cat_array = np.array(cat_values, dtype=np.float64).reshape(1, -1)

    # 2. Escalar variables numéricas
    num_df = df[config.NUMERICAL_COLUMNS].astype(np.float64)
    num_scaled = scaler.transform(num_df)  # scaler espera 7 características

    # 3. Concatenar en el orden: numéricas primero, luego categóricas
    X = np.hstack([num_scaled, cat_array])

    # Verificar dimensiones
    expected_features = binary_model.input_shape[1]  # debería ser 20
    if X.shape[1] != expected_features:
        raise ValueError(
            f"El número de características generadas ({X.shape[1]}) "
            f"no coincide con lo que espera el modelo ({expected_features})."
        )

    return X

# === ENDPOINTS DE PRUEBA ===

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Predicción de Riesgo Crediticio",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Documentación interactiva",
            "/ping": "Verificación simple",
            "/predict/binary": "Predicción binaria (Good/Bad)",
            "/predict/risk_level": "Predicción de nivel de riesgo",
            "/health": "Estado del servidor"
        }
    }

@app.get("/ping")
async def ping():
    """Endpoint simple para verificar conectividad"""
    return {"ping": "pong"}

@app.get("/health")
async def health_check():
    """Verifica el estado del servidor y modelos"""
    models_loaded = (binary_model is not None and
                     multiclass_model is not None and
                     scaler is not None and
                     encoders is not None)

    return {
        "status": "healthy" if models_loaded else "models_not_loaded",
        "binary_model": binary_model is not None,
        "multiclass_model": multiclass_model is not None,
        "scaler": scaler is not None,
        "encoders": encoders is not None,
        "message": "API funcionando correctamente" if models_loaded else "Faltan modelos o preprocesadores"
    }

# === ENDPOINTS DE PREDICCIÓN ===

@app.post("/predict/binary", response_model=BinaryPredictionResponse)
async def predict_binary(application: CreditApplication):
    """
    Predice si el crédito será bueno o malo
    """
    if binary_model is None:
        raise HTTPException(status_code=503, detail="Modelo binario no disponible. Verificar carga.")

    try:
        X = preprocess_input(application)
        proba = binary_model.predict(X, verbose=0)[0][0]
        prediction = "Bad" if proba > 0.5 else "Good"
        confidence = max(proba, 1 - proba)

        return BinaryPredictionResponse(
            prediction=prediction,
            probability_bad=float(proba),
            probability_good=float(1 - proba),
            confidence=float(confidence)
        )
    except Exception as e:
        print("="*60)
        print("ERROR en /predict/binary:")
        traceback.print_exc()
        print("="*60)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/predict/risk_level", response_model=RiskLevelResponse)
async def predict_risk_level(application: CreditApplication):
    """
    Predice el nivel de riesgo crediticio
    """
    if multiclass_model is None:
        raise HTTPException(status_code=503, detail="Modelo multiclase no disponible. Verificar carga.")

    try:
        X = preprocess_input(application)
        probas = multiclass_model.predict(X, verbose=0)[0]
        risk_idx = int(np.argmax(probas))
        risk_level = config.RISK_LEVELS[risk_idx]

        recommendations = {
            "Bajo": "Crédito aprobado con condiciones estándar",
            "Medio": "Crédito aprobado con seguimiento",
            "Alto": "Requiere garantías adicionales",
            "Crítico": "No recomendado aprobar"
        }

        return RiskLevelResponse(
            risk_level=risk_level,
            probabilities={
                level: float(prob)
                for level, prob in zip(config.RISK_LEVELS, probas)
            },
            recommendation=recommendations[risk_level]
        )
    except Exception as e:
        print("="*60)
        print("ERROR en /predict/risk_level:")
        traceback.print_exc()
        print("="*60)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# === EJECUTAR ===
if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor FastAPI...")
    print("Documentación: http://localhost:8000/docs")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, reload=config.API_RELOAD)
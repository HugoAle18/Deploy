import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURACIÓN DE PÁGINA (UI PROFESIONAL)
# ============================================================
st.set_page_config(
    page_title="Sistema de Alerta Temprana",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# MODELO
# ============================================================
model = joblib.load("modelo_final.pkl")

# ============================================================
# HEADER
# ============================================================
st.title("🎓 Sistema Inteligente de Predicción de Deserción")
st.markdown("### Modelo de Machine Learning para análisis académico")

st.divider()

# ============================================================
# LAYOUT PROFESIONAL (2 COLUMNAS)
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Rendimiento Académico")

    approved1 = st.number_input("Aprobados Semestre 1", 0.0, 30.0)
    grade1 = st.number_input("Nota Promedio Semestre 1", 0.0, 20.0)
    approved2 = st.number_input("Aprobados Semestre 2", 0.0, 30.0)
    grade2 = st.number_input("Nota Promedio Semestre 2", 0.0, 20.0)

    enrolled1 = st.number_input("Cursos inscritos S1", 0.0, 30.0)
    enrolled2 = st.number_input("Cursos inscritos S2", 0.0, 30.0)

with col2:
    st.subheader("👤 Información del Estudiante")

    age = st.number_input("Edad", 15, 80)
    gender = st.selectbox("Género", ["Femenino", "Masculino"])
    displaced = st.selectbox("Desplazado", ["No", "Sí"])

    st.subheader("💰 Situación Financiera")
    tuition = st.selectbox("Pagos al día", ["Sí", "No"])
    debtor = st.selectbox("Deudor", ["No", "Sí"])
    scholarship = st.selectbox("Becado", ["No", "Sí"])

    st.subheader("🌎 Contexto Económico")
    gdp = st.number_input("PBI")
    unemployment = st.number_input("Desempleo")
    inflation = st.number_input("Inflación")

# ============================================================
# BOTÓN DE PREDICCIÓN
# ============================================================
st.divider()

if st.button("🔍 Analizar Riesgo de Deserción", use_container_width=True):

    # ========================================================
    # PREPROCESAMIENTO (IGUAL AL ENTRENAMIENTO)
    # ========================================================
    data = {
        "Curricular units 1st sem (approved)": approved1,
        "Curricular units 1st sem (grade)": grade1,
        "Curricular units 2nd sem (approved)": approved2,
        "Curricular units 2nd sem (grade)": grade2,
        "Tuition fees up to date": 1 if tuition == "Sí" else 0,
        "Debtor": 1 if debtor == "Sí" else 0,
        "Scholarship holder": 1 if scholarship == "Sí" else 0,
        "Age at enrollment": age,
        "Displaced": 1 if displaced == "Sí" else 0,
        "Gender": 1 if gender == "Masculino" else 0,
        "GDP": gdp,
        "Unemployment rate": unemployment,
        "Inflation rate": inflation,
        "Curricular units 1st sem (enrolled)": enrolled1,
        "Curricular units 2nd sem (enrolled)": enrolled2,
    }

    # ========================================================
    # FEATURES DERIVADAS
    # ========================================================
    data["aprobacion_rate_1"] = approved1 / enrolled1 if enrolled1 > 0 else 0
    data["aprobacion_rate_2"] = approved2 / enrolled2 if enrolled2 > 0 else 0
    data["variacion_rendimiento"] = grade2 - grade1
    data["carga_total"] = enrolled1 + enrolled2
    data["riesgo_financiero"] = (
        (data["Tuition fees up to date"] == 0) +
        (data["Debtor"] == 1) +
        (data["Scholarship holder"] == 0)
    )

    df_input = pd.DataFrame([data])

    # ========================================================
    # PREDICCIÓN
    # ========================================================
    proba = model.predict_proba(df_input)[0][1]

    # ========================================================
    # RESULTADO VISUAL PROFESIONAL
    # ========================================================
    st.subheader("📊 Resultado del Análisis")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Probabilidad de Deserción", f"{proba:.2%}")

    with colB:
        if proba >= 0.7:
            st.error("RIESGO CRÍTICO")
        elif proba >= 0.4:
            st.warning("RIESGO MEDIO")
        else:
            st.success("RIESGO BAJO")

    with colC:
        st.info("Modelo ML - MLP Classifier")

    # ========================================================
    # BARRA VISUAL
    # ========================================================
    st.progress(float(proba))

    # ========================================================
    # RECOMENDACIÓN
    # ========================================================
    st.subheader("💡 Recomendación")

    if proba >= 0.7:
        st.write("⚠️ Intervención inmediata: tutoría + seguimiento académico + apoyo financiero")
    elif proba >= 0.4:
        st.write("📌 Monitoreo continuo y apoyo académico preventivo")
    else:
        st.write("✅ Estudiante en condición estable")

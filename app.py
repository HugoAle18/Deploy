import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Sistema de Alerta Temprana",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# CARGAR PIPELINE (CLAVE)
# ============================================================
model = joblib.load("modelo_final.pkl")  # 🔥 Pipeline completo

# ============================================================
# FEATURES (MISMO ORDEN QUE ENTRENAMIENTO)
# ============================================================
features_order = [
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Tuition fees up to date',
    'Debtor',
    'Scholarship holder',
    'Age at enrollment',
    'Displaced',
    'Gender',
    'GDP',
    'Unemployment rate',
    'Inflation rate',
    'aprobacion_rate_1',
    'aprobacion_rate_2',
    'variacion_rendimiento',
    'carga_total',
    'riesgo_financiero',
    'ratio_notas',
    'estres_academico'
]

# ============================================================
# UI
# ============================================================
st.title("🎓 Sistema Inteligente de Predicción de Deserción")
st.markdown("### Modelo de Machine Learning (MLP + Pipeline)")
st.divider()

col1, col2 = st.columns(2)

# =============================
# COLUMNA 1
# =============================
with col1:
    st.subheader("📚 Rendimiento Académico")

    approved1 = st.number_input("Aprobados S1", 0.0, 30.0)
    grade1 = st.number_input("Nota S1", 0.0, 20.0)
    approved2 = st.number_input("Aprobados S2", 0.0, 30.0)
    grade2 = st.number_input("Nota S2", 0.0, 20.0)

    enrolled1 = st.number_input("Cursos inscritos S1", 0.0, 30.0)
    enrolled2 = st.number_input("Cursos inscritos S2", 0.0, 30.0)

# =============================
# COLUMNA 2
# =============================
with col2:
    st.subheader("👤 Datos del Estudiante")

    age = st.number_input("Edad", 15, 80)
    gender = st.selectbox("Género", ["Femenino", "Masculino"])
    displaced = st.selectbox("Desplazado", ["No", "Sí"])

    st.subheader("💰 Finanzas")
    tuition = st.selectbox("Pago al día", ["Sí", "No"])
    debtor = st.selectbox("Deudor", ["No", "Sí"])
    scholarship = st.selectbox("Becado", ["No", "Sí"])

    st.subheader("🌎 Economía")
    gdp = st.number_input("PBI")
    unemployment = st.number_input("Desempleo")
    inflation = st.number_input("Inflación")

# ============================================================
# BOTÓN
# ============================================================
st.divider()

if st.button("🔍 Analizar Riesgo", use_container_width=True):

    # ========================================================
    # DATA BASE
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
    }

    # ========================================================
    # FEATURES DERIVADAS (IGUAL QUE ENTRENAMIENTO)
    # ========================================================
    aprobacion_rate_1 = approved1 / enrolled1 if enrolled1 > 0 else 0
    aprobacion_rate_2 = approved2 / enrolled2 if enrolled2 > 0 else 0
    variacion_rendimiento = grade2 - grade1
    carga_total = enrolled1 + enrolled2
    riesgo_financiero = (
        (1 if tuition == "No" else 0) +
        (1 if debtor == "Sí" else 0) +
        (1 if scholarship == "No" else 0)
    )

    ratio_notas = grade2 / (grade1 + 1e-5)
    estres_academico = carga_total / (age + 1)

    # Agregar al diccionario
    data.update({
        "aprobacion_rate_1": aprobacion_rate_1,
        "aprobacion_rate_2": aprobacion_rate_2,
        "variacion_rendimiento": variacion_rendimiento,
        "carga_total": carga_total,
        "riesgo_financiero": riesgo_financiero,
        "ratio_notas": ratio_notas,
        "estres_academico": estres_academico
    })

    # ========================================================
    # DATAFRAME FINAL ORDENADO
    # ========================================================
    df_input = pd.DataFrame([data])
    df_input = df_input[features_order]  # 🔥 ORDEN EXACTO

    # ========================================================
    # PREDICCIÓN (PIPELINE)
    # ========================================================
    proba = model.predict_proba(df_input)[0][1]

    # ========================================================
    # RESULTADO
    # ========================================================
    st.subheader("📊 Resultado")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Probabilidad", f"{proba:.2%}")

    with colB:
        if proba >= 0.7:
            st.error("RIESGO CRÍTICO")
        elif proba >= 0.4:
            st.warning("RIESGO MEDIO")
        else:
            st.success("RIESGO BAJO")

    with colC:
        st.info("Modelo MLP + Pipeline")

    st.progress(float(proba))

    # ========================================================
    # RECOMENDACIÓN
    # ========================================================
    st.subheader("💡 Recomendación")

    if proba >= 0.7:
        st.write("⚠️ Intervención inmediata (académica + financiera)")
    elif proba >= 0.4:
        st.write("📌 Seguimiento continuo")
    else:
        st.write("✅ Estudiante estable")

    # ========================================================
    # DEBUG (MUY IMPORTANTE PARA VALIDAR)
    # ========================================================
    with st.expander("🔎 Ver datos procesados"):
        st.dataframe(df_input)

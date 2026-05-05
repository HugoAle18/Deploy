import streamlit as st
import joblib
import pandas as pd

# ============================================================
# CONFIGURACIÓN
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
# ORDEN DE FEATURES (NO TOCAR)
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
# HEADER
# ============================================================
st.title("🎓 Sistema Inteligente de Predicción de Deserción")

st.markdown("""
### 🧠 ¿Cómo usar este sistema?

Ingrese los datos del estudiante.

📌 **Ejemplo de estudiante promedio:**
- Aprobados: 4 de 6 cursos  
- Notas: 13 → 12  
- Edad: 20  
- Pagos al día: Sí  

👉 Puedes usar esos valores para probar.
""")

st.info("""
🎯 CASOS DE PRUEBA:

🟢 Bajo riesgo:
- 5/6 cursos aprobados | Notas: 15 → 14  

🟡 Riesgo medio:
- 4/6 cursos | Notas: 13 → 12  

🔴 Riesgo alto:
- 2/6 cursos | Notas: 11 → 10  
""")

st.divider()

# ============================================================
# LAYOUT
# ============================================================
col1, col2 = st.columns(2)

# ============================================================
# ACADÉMICO
# ============================================================
with col1:
    st.subheader("📚 Rendimiento Académico")

    approved1 = st.number_input(
        "Aprobados S1",
        0.0, 30.0, value=4.0,
        help="Ejemplo: 4 cursos aprobados de 6"
    )

    grade1 = st.number_input(
        "Nota S1",
        0.0, 20.0, value=13.0,
        help="10-14 regular | 15+ buen rendimiento"
    )

    approved2 = st.number_input("Aprobados S2", 0.0, 30.0, value=4.0)
    grade2 = st.number_input("Nota S2", 0.0, 20.0, value=12.0)

    enrolled1 = st.number_input("Cursos inscritos S1", 1.0, 30.0, value=6.0)
    enrolled2 = st.number_input("Cursos inscritos S2", 1.0, 30.0, value=6.0)

# ============================================================
# PERSONAL + FINANZAS
# ============================================================
with col2:
    st.subheader("👤 Datos del Estudiante")

    age = st.number_input("Edad", 15, 80, value=20)

    gender = st.selectbox("Género", ["Femenino", "Masculino"])

    displaced = st.selectbox(
        "Desplazado",
        ["No", "Sí"],
        help="Si el estudiante cambió de ciudad/región"
    )

    st.subheader("💰 Situación Financiera")

    tuition = st.selectbox(
        "Pagos al día",
        ["Sí", "No"],
        help="Pagar a tiempo reduce el riesgo"
    )

    debtor = st.selectbox("Deudor", ["No", "Sí"])

    scholarship = st.selectbox(
        "Becado",
        ["No", "Sí"],
        help="Tener beca reduce el riesgo"
    )

# ============================================================
# ECONOMÍA (OCULTA)
# ============================================================
with st.expander("🌎 Factores Económicos (Opcional)"):
    st.markdown("Puedes dejar estos valores por defecto")

    gdp = st.number_input("PBI", value=1.5)
    unemployment = st.number_input("Desempleo (%)", value=10.0)
    inflation = st.number_input("Inflación (%)", value=2.0)

# ============================================================
# BOTÓN
# ============================================================
st.divider()

if st.button("🔍 Analizar Riesgo", use_container_width=True):

    # =============================
    # FEATURES DERIVADAS
    # =============================
    aprobacion_rate_1 = approved1 / enrolled1
    aprobacion_rate_2 = approved2 / enrolled2
    variacion_rendimiento = grade2 - grade1
    carga_total = enrolled1 + enrolled2

    riesgo_financiero = (
        (1 if tuition == "No" else 0) +
        (1 if debtor == "Sí" else 0) +
        (1 if scholarship == "No" else 0)
    )

    ratio_notas = grade2 / (grade1 + 1e-5)
    estres_academico = carga_total / (age + 1)

    # =============================
    # DATA FINAL
    # =============================
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
        "aprobacion_rate_1": aprobacion_rate_1,
        "aprobacion_rate_2": aprobacion_rate_2,
        "variacion_rendimiento": variacion_rendimiento,
        "carga_total": carga_total,
        "riesgo_financiero": riesgo_financiero,
        "ratio_notas": ratio_notas,
        "estres_academico": estres_academico
    }

    df_input = pd.DataFrame([data])[features_order]

    # =============================
    # PREDICCIÓN
    # =============================
    proba = model.predict_proba(df_input)[0][1]

    # =============================
    # RESULTADO
    # =============================
    st.subheader("📊 Resultado")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Probabilidad de deserción", f"{proba:.2%}")

    with colB:
        if proba >= 0.7:
            st.error("🔴 RIESGO CRÍTICO")
        elif proba >= 0.4:
            st.warning("🟡 RIESGO MEDIO")
        else:
            st.success("🟢 RIESGO BAJO")

    with colC:
        st.info("Modelo: MLP + Pipeline")

    st.progress(float(proba))

    # =============================
    # INTERPRETACIÓN SIMPLE
    # =============================
    st.subheader("💡 Interpretación")

    if proba >= 0.7:
        st.write("⚠️ Alto riesgo: bajo rendimiento o problemas financieros")
    elif proba >= 0.4:
        st.write("📌 Riesgo medio: monitorear rendimiento")
    else:
        st.write("✅ Estudiante estable")

    # =============================
    # DEBUG
    # =============================
    with st.expander("🔎 Ver datos procesados"):
        st.dataframe(df_input)

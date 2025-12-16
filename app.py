import streamlit as st
import pandas as pd
import pickle

# ==================================================
# PAGE CONFIG (HARUS PALING ATAS)
# ==================================================
st.set_page_config(
    page_title="Telemarketing Lead Scoring",
    page_icon="📞",
    layout="centered"
)

# ==================================================
# MODEL FEATURES (WAJIB SAMA PERSIS DENGAN TRAINING)
# ==================================================
MODEL_FEATURES = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous',
    'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    with open("model_new.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ==================================================
# TITLE
# ==================================================
st.title("📞 Telemarketing Lead Scoring")
st.write("Prediksi kemungkinan nasabah **subscribe Term Deposit**")
st.divider()

# ==================================================
# INPUT NASABAH
# ==================================================
st.subheader("🧾 Data Nasabah")

# ---- Numeric ----
age = st.number_input("Usia", min_value=18, max_value=100, value=35)

# ---- Job ----
job_label_to_value = {
    "Admin": "admin.",
    "Blue Collar": "blue-collar",
    "Technician": "technician",
    "Services": "services",
    "Management": "management",
    "Retired": "retired",
    "Entrepreneur": "entrepreneur",
    "Self Employed": "self-employed",
    "Student": "student",
    "Unemployed": "unemployed",
    "Unknown": "unknown"
}
job_label = st.selectbox("Pekerjaan", list(job_label_to_value.keys()))
job = job_label_to_value[job_label]

# ---- Marital ----
marital_map = {
    "Married": "married",
    "Single": "single",
    "Divorced": "divorced"
}
marital_label = st.selectbox("Marital Status", list(marital_map.keys()))
marital = marital_map[marital_label]

# ---- Education ----
education_map = {
    "Basic 4y": "basic.4y",
    "Basic 6y": "basic.6y",
    "Basic 9y": "basic.9y",
    "High School": "high.school",
    "Professional Course": "professional.course",
    "University Degree": "university.degree",
    "Unknown": "unknown"
}
education_label = st.selectbox("Education", list(education_map.keys()))
education = education_map[education_label]

# ---- Loans ----
housing = st.selectbox("Housing Loan", ["No", "Yes"]).lower()
loan = st.selectbox("Personal Loan", ["No", "Yes"]).lower()

# ==================================================
# DATAFRAME INPUT (STRICT MODEL FORMAT)
# ==================================================
input_df = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": "no",
    "housing": housing,
    "loan": loan,
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191
}])

# 🔒 DROP KOLOM LIAR + KUNCI URUTAN
input_df = input_df[MODEL_FEATURES]

# ==================================================
# PREDICTION
# ==================================================
st.divider()

if st.button("🔍 Predict"):
    pred_label = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")

    if pred_label == 1:
        st.success(f"✅ **Subscribe**\n\nProbability: **{pred_prob:.2%}**")
    else:
        st.error(f"❌ **Not Subscribe**\n\nProbability: **{pred_prob:.2%}**")

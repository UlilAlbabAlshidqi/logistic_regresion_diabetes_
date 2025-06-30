import pickle 
import streamlit as st
import joblib
import numpy as np

# Membaca model dan scaler
diabetes_model = joblib.load("model_svm_pkl")
scaler = joblib.load("scaler_pkl")

# Judul Web
st.title("Data Mining Prediksi Diabetes")

# Form Input Bagi Kolom
col1, col2 = st.columns(2)

with col1: 
    Pregnancies = st.number_input("Input Nilai Pregnancies", step=1, format="%d")
with col2:
    Glucose = st.number_input("Input Nilai Glucose", step=1, format="%d")
with col1: 
    BloodPressure = st.number_input("Input Nilai Blood Pressure", step=1, format="%d")
with col2:
    SkinThickness = st.number_input("Input Nilai Skin Thickness", step=1, format="%d")
with col1: 
    Insulin = st.number_input("Input Nilai Insulin", step=1, format="%d")
with col2:
    BMI = st.number_input("Input Nilai BMI")
with col1:
    DiabetesPedigreeFunction = st.number_input("Input Nilai Diabetes Pedigree Function")
with col2:
    Age = st.number_input("Input Nilai Age", step=1, format="%d")

# Tombol prediksi
if st.button("Prediksi"):
    # Transform log
    Insulin_log = np.log1p(Insulin)
    DPF_log = np.log1p(DiabetesPedigreeFunction)

    # Susun input sesuai urutan fitur model:
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age', 'Insulin_log', 'DPF_log']
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, Age, Insulin_log, DPF_log]])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    pred = diabetes_model.predict(input_scaled)
    prob = diabetes_model.predict_proba(input_scaled)

    # Output
    if pred[0] == 0:
        st.success("✅ Pasien Tidak Terkena Diabetes")
    else:
        st.error("⚠️ Pasien Terkena Diabetes")

    st.write(f'Probabilitas terkena diabetes: **{prob[0][1]:.2f}**')
    st.caption("Hasil ini bersifat prediktif, bukan diagnosis medis. Konsultasikan lebih lanjut dengan dokter.")

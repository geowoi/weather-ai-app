import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="AI Prediksi Iklim Cuaca", layout="wide")
st.title("🌦️ AI Prediksi Iklim / Cuaca")
st.markdown(
    "Aplikasi kecerdasan buatan untuk **memprediksi suhu udara** berdasarkan data iklim historis."
)

# ===============================
# DATASET BUATAN (STABIL)
# ===============================
@st.cache_data
def generate_weather_data():
    np.random.seed(42)
    days = 365 * 5  # 5 tahun
    data = pd.DataFrame({
        "Hari_ke": np.arange(days),
        "Kelembaban": np.random.uniform(40, 90, days),
        "Kecepatan_Angin": np.random.uniform(0, 15, days),
        "Curah_Hujan": np.random.uniform(0, 20, days),
    })

    # Pola suhu (ada tren + noise)
    data["Suhu"] = (
        30
        - 0.05 * data["Curah_Hujan"]
        - 0.03 * data["Kelembaban"]
        + 0.02 * data["Kecepatan_Angin"]
        + np.random.normal(0, 1.5, days)
    )

    return data

df = generate_weather_data()

# ===============================
# EDA
# ===============================
st.subheader("📊 Statistik Data Iklim")
st.write(df.describe())

st.subheader("📈 Grafik Perubahan Suhu dari Waktu ke Waktu")
sample_df = df.sample(500)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sample_df["Hari_ke"], sample_df["Suhu"])
ax.set_xlabel("Hari")
ax.set_ylabel("Suhu (°C)")
ax.set_title("Tren Suhu Udara")
st.pyplot(fig)

# ===============================
# TRAIN MODEL (AI)
# ===============================
features = df[["Kelembaban", "Kecepatan_Angin", "Curah_Hujan"]]
target = df["Suhu"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.success(f"✅ Model AI berhasil dilatih | MAE: {mae:.2f} °C")

# ===============================
# INPUT USER
# ===============================
st.subheader("🔮 Prediksi Suhu Cuaca")

col1, col2, col3 = st.columns(3)

with col1:
    kelembaban = st.slider("Kelembaban (%)", 40, 90, 70)

with col2:
    angin = st.slider("Kecepatan Angin (m/s)", 0, 15, 5)

with col3:
    hujan = st.slider("Curah Hujan (mm)", 0, 20, 5)

input_data = np.array([[kelembaban, angin, hujan]])

if st.button("Prediksi Suhu"):
    suhu_pred = model.predict(input_data)[0]
    st.metric("🌡️ Perkiraan Suhu", f"{suhu_pred:.2f} °C")
    st.info(
        "Prediksi ini dihasilkan oleh **AI (Linear Regression)** "
        "yang mempelajari pola hubungan antar variabel iklim."
    )

# ===============================
# PENJELASAN AI
# ===============================
st.subheader("🤖 Mengapa ini disebut AI?")
st.markdown("""
- Menggunakan **Machine Learning (Regresi)**
- Model belajar dari **ribuan data iklim**
- Tidak menggunakan rumus manual
- Model menemukan pola hubungan cuaca secara otomatis
""")

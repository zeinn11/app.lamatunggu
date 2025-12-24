import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Prediksi Lama Tunggu Kerja", layout="centered")

st.title("📊 Analisis & Prediksi Lama Tunggu Kerja Alumni")

# =========================
# LOAD DATA
# =========================
@st.cache_data (ttl=60)
def load_data():
    df = pd.read_excel("tracer_studi.xlsx")
    df = df[df["Tahun_lulus"] >= 2010]

    def convert_wait(x):
        if pd.isna(x):
            return np.nan
        x = str(x).lower()
        if "0" in x:
            return 3
        if "6" in x and "12" in x:
            return 9
        if ">12" in x:
            return 15
        return np.nan

    df["Lama_tunggu_num"] = df["Lama_tunggu_kerja"].apply(convert_wait)

    ts = df.groupby("Tahun_lulus")["Lama_tunggu_num"].mean()
    ts = ts.sort_index().interpolate()

    return ts

ts = load_data()
# FILTER DATA MULAI 2018
df = df[df["Tahun_lulus"] >= 2018]

# =========================
# TAMPILKAN DATA AKTUAL
# =========================
st.subheader("📌 Data Aktual Lama Tunggu Kerja (rata-rata per tahun)")
st.dataframe(ts)

# =========================
# MODEL ARIMA
# =========================
model = ARIMA(ts, order=(1, 0, 1))
model_fit = model.fit()

# =========================
# PREDIKSI 2026
# =========================
forecast_2026 = model_fit.forecast(steps=1)

st.subheader("🔮 Prediksi Lama Tunggu Kerja Tahun 2026")
st.metric(
    label="Rata-rata Lama Tunggu (bulan)",
    value=f"{forecast_2026.iloc[0]:.2f} bulan"
)

# =========================
# GRAFIK
# =========================
st.subheader("📈 Grafik Lama Tunggu Kerja")

fig, ax = plt.subplots()
ax.plot(ts.index, ts.values, marker="o", label="Data Aktual")
ax.axhline(
    forecast_2026.iloc[0],
    linestyle="--",
    color="red",
    label="Prediksi 2026"
)

ax.set_xlabel("Tahun Lulus")
ax.set_ylabel("Bulan")
ax.legend()

st.pyplot(fig)



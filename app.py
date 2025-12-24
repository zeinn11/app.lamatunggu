import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Prediksi Lama Tunggu Kerja", layout="centered")

st.title("📊 Prediksi Lama Tunggu Kerja Alumni")

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
    ts = ts.interpolate()

    return ts

ts = load_data()

st.subheader("📈 Data Aktual Lama Tunggu Kerja")
st.line_chart(ts)

# ===== MODEL =====
model = ARIMA(ts, order=(1, 0, 1))
model_fit = model.fit()

forecast_year = st.slider("Pilih tahun prediksi", 2026, 2030, 2026)
steps = forecast_year - ts.index.max()

forecast = model_fit.forecast(steps=steps)

st.subheader(f"🔮 Prediksi Lama Tunggu Kerja Tahun {forecast_year}")
st.success(f"Rata-rata: **{forecast.iloc[-1]:.2f} bulan**")

# ===== PLOT =====
fig, ax = plt.subplots()
ts.plot(ax=ax, label="Aktual")
forecast.index = range(ts.index.max()+1, forecast_year+1)
forecast.plot(ax=ax, label="Prediksi", linestyle="--")
ax.legend()
st.pyplot(fig)

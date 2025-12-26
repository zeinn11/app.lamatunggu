import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Forecast Lama Tunggu Kerja", layout="centered")

st.title("📊 Prediksi Lama Tunggu Kerja Alumni")
st.write("Menggunakan Time Series ARIMA")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_excel("tracer_studi.xlsx")
    return df

df = load_data()
st.subheader("Data Awal")
st.dataframe(df.head())

# ======================
# PREPROCESSING
# ======================
def convert_income(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower()
    if "1-3" in x:
        return 2000000
    if "3-5" in x:
        return 4000000
    if ">5" in x:
        return 6000000
    return np.nan

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

df["Pendapatan_num"] = df["Pendapatan_bersih"].apply(convert_income)
df["Lama_tunggu_num"] = df["Lama_tunggu_kerja"].apply(convert_wait)

# Filter tahun >= 2018
df = df[df["Tahun_lulus"] >= 2018]

# ======================
# TIME SERIES
# ======================
ts = df.groupby("Tahun_lulus")["Lama_tunggu_num"].mean()
ts = ts.sort_index()
ts = ts.interpolate()
ts.index = ts.index.astype(int)

st.subheader("Rata-rata Lama Tunggu per Tahun")
st.write(ts)

# ======================
# ADF TEST
# ======================
adf = adfuller(ts)
st.subheader("Uji Stasioneritas (ADF Test)")
st.write(f"ADF Statistic: {adf[0]:.4f}")
st.write(f"p-value: {adf[1]:.4f}")

# ======================
# TRAIN TEST SPLIT
# ======================
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# ======================
# MODEL ARIMA
# ======================
model = ARIMA(train, order=(1,0,1))
model_fit = model.fit()

pred = model_fit.forecast(steps=len(test))
pred.index = test.index

mae = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))

st.subheader("Evaluasi Model")
st.write(f"MAE  : {mae:.2f}")
st.write(f"RMSE : {rmse:.2f}")

# ======================
# FINAL MODEL & FORECAST 2026
# ======================
final_model = ARIMA(ts, order=(1,0,1))
final_fit = final_model.fit()

forecast_2026 = final_fit.forecast(steps=1)
tahun_2026 = ts.index.max() + 1

st.subheader("Prediksi Tahun 2026")
st.write(f"Tahun {tahun_2026}: **{forecast_2026.iloc[0]:.2f} bulan**")

# ======================
# VISUALISASI
# ======================
ts_plot = ts.copy()
ts_plot.loc[tahun_2026] = forecast_2026.iloc[0]

fig, ax = plt.subplots()
ax.plot(ts_plot.index, ts_plot.values, marker="o")
ax.set_title("Forecast Lama Tunggu Kerja Alumni")
ax.set_xlabel("Tahun")
ax.set_ylabel("Lama Tunggu (Bulan)")
ax.grid(True)

st.pyplot(fig)

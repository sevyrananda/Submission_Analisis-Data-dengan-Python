import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# --- LOAD DATA ---
df_day = pd.read_csv("./dataset/day.csv")
df_hour = pd.read_csv("./dataset/hour.csv")

# Konversi kolom tanggal ke format datetime
df_day['dteday'] = pd.to_datetime(df_day['dteday'])
df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])

# Mapping Musim
season_mapping = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
df_day["season_label"] = df_day["season"].map(season_mapping)

# Kategori Hari Kerja vs Akhir Pekan
df_day["weekday_category"] = df_day["weekday"].apply(lambda x: "Weekend" if x in [0, 6] else "Weekday")

# --- TITLE ---
st.title("🚲 **Dashboard Interaktif Bike Sharing**")
st.write("📌 Dashboard ini memahami pola peminjaman sepeda berdasarkan musim, suhu, waktu, dan kategori hari.")

# --- IDENTITAS ---
st.sidebar.markdown("## Identitas")
st.sidebar.markdown("**👩🏻‍💻 Nama:** Sevyra Nanda Octavianti  \n"
                    "**📧 Email:** sevyra02@gmail.com  \n"
                    "**🆔 ID Dicoding:** sevyrananda")

# --- FILTER DATA ---
st.sidebar.header("🎛️ **Filter Data**")
selected_season = st.sidebar.selectbox("Pilih Musim", ["All"] + list(df_day["season_label"].unique()))
selected_day_type = st.sidebar.radio("Pilih Kategori Hari:", ["All", "Weekday", "Weekend"])
temp_range = st.sidebar.slider("Filter Suhu (Normalized)", 0.0, 1.0, (0.2, 0.8))
show_regression = st.sidebar.checkbox("Tampilkan Prediksi Linear")

df_filtered = df_day.copy()
if selected_season != "All":
    df_filtered = df_filtered[df_filtered["season_label"] == selected_season]
if selected_day_type != "All":
    df_filtered = df_filtered[df_filtered["weekday_category"] == selected_day_type]
df_filtered = df_filtered[(df_filtered["temp"] >= temp_range[0]) & (df_filtered["temp"] <= temp_range[1])]

# --- INSIGHT PANEL ---
st.subheader("🔍 Key Insights")
col1, col2 = st.columns(2)

with col1:
    st.info("🌦️ **Musim dengan peminjaman tertinggi:**\n"  
            "\n" 
            f"🏆 {df_day.groupby('season_label')['cnt'].mean().idxmax()}")

with col2:
    st.success("🌡️ **Suhu optimal peminjaman tertinggi:**\n"  
               "\n"
               f"🔥 {df_day.groupby('temp')['cnt'].mean().idxmax():.2f} (normalized)")

# --- VISUALISASI 1: Peminjaman Sepeda Berdasarkan Musim ---
st.subheader("📊 **Peminjaman Sepeda Berdasarkan Musim**")
st.write("**Insight:** Musim tertentu memiliki dampak signifikan pada jumlah peminjaman sepeda. Kita dapat melihat bahwa musim Fall memiliki jumlah peminjaman tertinggi.")

fig1 = px.bar(df_filtered, x="season_label", y="cnt", color="season_label",
              labels={"season_label": "Musim", "cnt": "Jumlah Peminjaman"},
              title="Jumlah Peminjaman Berdasarkan Musim",
              hover_name="season_label")
st.plotly_chart(fig1, use_container_width=True)

# --- VISUALISASI 2: Hubungan Suhu dengan Peminjaman Sepeda ---
st.subheader("🌡️ **Hubungan Suhu dengan Peminjaman Sepeda**")
st.write("**Insight:** Semakin tinggi suhu (dalam skala normalisasi), semakin tinggi peminjaman sepeda, dengan pola tertentu yang dapat kita amati.")

fig2 = px.scatter(df_filtered, x="temp", y="cnt",
                  labels={"temp": "Suhu", "cnt": "Jumlah Peminjaman"},
                  title="Pengaruh Suhu terhadap Peminjaman Sepeda",
                  color="temp", hover_data=["season_label"])
st.plotly_chart(fig2, use_container_width=True)

# --- VISUALISASI 3: Peminjaman Sepeda Berdasarkan Jam ---
st.subheader("⏰ Peminjaman Sepeda Berdasarkan Jam")
st.write("**Insight:** Peminjaman sepeda cenderung meningkat di jam-jam sibuk (pagi dan sore hari), yang kemungkinan besar karena aktivitas komuter.")

# Slider untuk memilih rentang jam
hour_range = st.slider("Pilih Rentang Jam:", 0, 23, (6, 18))

# Filter data sesuai rentang jam
df_hour_filtered = df_hour[(df_hour["hr"] >= hour_range[0]) & (df_hour["hr"] <= hour_range[1])]

# **Agregasi data: rata-rata jumlah peminjaman per jam**
df_hour_avg = df_hour_filtered.groupby("hr", as_index=False)["cnt"].mean()

# Line chart interaktif dengan Plotly
fig3 = px.line(df_hour_avg, x="hr", y="cnt",
               labels={"hr": "Jam", "cnt": "Rata-rata Peminjaman"},
               title="Peminjaman Sepeda Berdasarkan Jam",
               markers=True,  # Menambahkan titik pada tiap data
               line_shape="linear")  # Pastikan bentuk garis linear

st.plotly_chart(fig3)


# --- VISUALISASI 4: Hari Kerja vs Akhir Pekan ---
st.subheader("📅 Peminjaman Sepeda: Hari Kerja vs Akhir Pekan")
st.write("**Insight:** Peminjaman lebih tinggi pada hari kerja, yang menunjukkan bahwa sepeda sering digunakan untuk transportasi kerja.")

fig4 = px.box(df_filtered, x="weekday_category", y="cnt", color="weekday_category",
              labels={"weekday_category": "Kategori Hari", "cnt": "Jumlah Peminjaman"},
              title="Distribusi Peminjaman Sepeda: Weekday vs Weekend")
st.plotly_chart(fig4)

# --- VISUALISASI 5: Distribusi Kategori Peminjaman ---
st.subheader("📦 Distribusi Kategori Peminjaman")
st.write("**Insight:** Mayoritas hari memiliki jumlah peminjaman dalam kategori 'Medium' dan 'High', dengan hanya sedikit hari dalam kategori 'Very High'.")

df_filtered["demand_category"] = pd.cut(df_filtered["cnt"], bins=[0, 2000, 4000, 6000, df_filtered["cnt"].max()], 
                                        labels=["Low", "Medium", "High", "Very High"])

fig5 = px.histogram(df_filtered, x="demand_category", color="demand_category",
                    labels={"demand_category": "Kategori Peminjaman", "cnt": "Jumlah Hari"},
                    title="Distribusi Kategori Peminjaman Sepeda")
st.plotly_chart(fig5)

# --- VISUALISASI 6: Heatmap Jam vs Hari ---
st.subheader("⏰ **Heatmap Peminjaman Sepeda (Jam vs Hari)**")
st.write("**Insight:** Heatmap ini menunjukkan jam-jam sibuk dalam seminggu. Terlihat bahwa peminjaman tertinggi terjadi di pagi dan sore hari pada hari kerja.")

df_hour_filtered = df_hour.pivot_table(values="cnt", index="weekday", columns="hr", aggfunc="mean")
fig3 = px.imshow(df_hour_filtered, color_continuous_scale="viridis",
                 labels=dict(color="Jumlah Peminjaman"),
                 title="Heatmap Peminjaman Sepeda Berdasarkan Jam & Hari")
st.plotly_chart(fig3, use_container_width=True)

# --- VISUALISASI 7: Tren Peminjaman Sepeda Sepanjang Tahun ---
st.subheader("📈 **Tren Peminjaman Sepeda Sepanjang Tahun**")
st.write("**Insight:** Tren peminjaman meningkat pada bulan tertentu. Ini dapat memberikan wawasan kapan puncak peminjaman terjadi.")

df_day["month"] = df_day["dteday"].dt.month
monthly_rentals = df_day.groupby("month")["cnt"].mean()
fig4 = px.line(x=monthly_rentals.index, y=monthly_rentals.values,
               labels={"x": "Bulan", "y": "Rata-rata Peminjaman"},
               title="Tren Bulanan Peminjaman Sepeda",
               markers=True)
st.plotly_chart(fig4, use_container_width=True)

# --- VISUALISASI 8: Prediksi Peminjaman Sepeda ---
if show_regression:
    st.subheader("🔮 **Prediksi Peminjaman Sepeda Berdasarkan Suhu**")
    st.write("**Insight:** Model regresi linier digunakan untuk memprediksi hubungan antara suhu dan jumlah peminjaman sepeda.")
    
    X = df_day[["temp"]]
    y = df_day["cnt"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    pred_temp = np.linspace(0, 1, 100).reshape(-1, 1)
    pred_cnt = model.predict(pred_temp)
    
    fig5 = px.scatter(df_day, x="temp", y="cnt", opacity=0.5, title="Prediksi Linear: Suhu vs Peminjaman")
    fig5.add_scatter(x=pred_temp.flatten(), y=pred_cnt, mode="lines", name="Regresi Linear", line=dict(color="red"))
    st.plotly_chart(fig5, use_container_width=True)

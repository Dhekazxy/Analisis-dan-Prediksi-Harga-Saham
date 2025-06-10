import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Prediksi Saham", layout="wide")

# --- CSS UNTUK TEMA GELAP ---
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #0D1B2A !important;
        color: white !important;
    }
    :root { color-scheme: dark !important; }
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stToolbar"], 
    [data-testid="stSidebar"], 
    .main, .block-container {
        background-color: #0D1B2A !important;
        color: white !important;
    }
    input, textarea, select {
        background-color: #1B263B !important;
        color: white !important;
        border: none !important;
    }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #1B263B; }
    ::-webkit-scrollbar-thumb { background: #415A77; }
    </style>
""", unsafe_allow_html=True)

# --- INISIALISASI SESSION STATE ---
# PERBAIKAN: Menambah 'page' untuk mengingat halaman aktif
if 'page' not in st.session_state:
    st.session_state.page = 'Evaluasi' # Halaman default

if 'page' not in st.session_state:
    st.session_state.page = 'Evaluasi'

if 'prediksi_dijalankan' not in st.session_state:
    st.session_state.prediksi_dijalankan = False
if 'df_hasil_prediksi' not in st.session_state:
    st.session_state.df_hasil_prediksi = None
if 'metode_terpilih' not in st.session_state:
    st.session_state.metode_terpilih = "LSTM - 30 hari - 4 layer - 100 epoch"

if 'evaluasi_dijalankan' not in st.session_state:
    st.session_state.evaluasi_dijalankan = False
if 'metrik_evaluasi' not in st.session_state:
    st.session_state.metrik_evaluasi = None
if 'eval_file_found' not in st.session_state:
    st.session_state.eval_file_found = False

# --- SIDEBAR NAVIGASI ---
# PERBAIKAN: Tombol sidebar sekarang mengubah 'st.session_state.page'
st.sidebar.title("📌 Navigasi")
if st.sidebar.button("📊 Evaluasi"):
    st.session_state.page = "Evaluasi"
if st.sidebar.button("🏠 Menu Utama"):
    st.session_state.page = "Menu Utama"
if st.sidebar.button("ℹ️ About"):
    st.session_state.page = "About"

# --- FUNGSI & DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("bbri.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df
df = load_data()
df_full = pd.read_csv("bbri.csv", parse_dates=['Date'], index_col='Date')
ts_full = df_full['Close']

MODEL_LIST = {
    "LSTM - 30 hari - 4 layer - 50 epoch": ("LSTM_30_4_50.keras", "LSTM_30_4_50_scaler.pkl"),
    "LSTM - 30 hari - 4 layer - 100 epoch    🥉": ("LSTM_30_4_100.keras", "LSTM_30_4_100_scaler.pkl"),
    "LSTM - 30 hari - 6 layer - 50 epoch": ("LSTM_30_6_50.keras", "LSTM_30_6_50_scaler.pkl"),
    "LSTM - 30 hari - 6 layer - 100 epoch": ("LSTM_30_6_100.keras", "LSTM_30_6_100_scaler.pkl"),
    "LSTM - 30 hari - 8 layer - 50 epoch": ("LSTM_30_8_50.keras", "LSTM_30_8_50_scaler.pkl"),
    "LSTM - 30 hari - 8 layer - 100 epoch": ("LSTM_30_8_100.keras", "LSTM_30_8_100_scaler.pkl"),
    "LSTM - 60 hari - 4 layer - 50 epoch": ("LSTM_60_4_50.keras", "LSTM_60_4_50_scaler.pkl"),
    "LSTM - 60 hari - 4 layer - 100 epoch    🥇": ("LSTM_60_4_100.keras", "LSTM_60_4_100_scaler.pkl"),
    "LSTM - 60 hari - 6 layer - 50 epoch": ("LSTM_60_6_50.keras", "LSTM_60_6_50_scaler.pkl"),
    "LSTM - 60 hari - 6 layer - 100 epoch    🥈": ("LSTM_60_6_100.keras", "LSTM_60_6_100_scaler.pkl"),
    "LSTM - 60 hari - 8 layer - 50 epoch": ("LSTM_60_8_50.keras", "LSTM_60_8_50_scaler.pkl"),
    "LSTM - 60 hari - 8 layer - 100 epoch": ("LSTM_60_8_100.keras", "LSTM_60_8_100_scaler.pkl"),
    "RNN - 30 hari - 100 layer - 50 epoch": ("RNN_30_100_50.keras", "RNN_30_100_50_scaler.pkl"),
    "RNN - 30 hari - 100 layer - 100 epoch": ("RNN_30_100_100.keras", "RNN_30_100_100_scaler.pkl"),
    "RNN - 30 hari - 150 layer - 50 epoch": ("RNN_30_150_50.keras", "RNN_30_150_50_scaler.pkl"),
    "RNN - 30 hari - 150 layer - 100 epoch": ("RNN_30_150_100.keras", "RNN_30_150_100_scaler.pkl"),
    "RNN - 60 hari - 100 layer - 50 epoch": ("RNN_60_100_50.keras", "RNN_60_100_50_scaler.pkl"),
    "RNN - 60 hari - 100 layer - 100 epoch": ("RNN_60_100_100.keras", "RNN_60_100_100_scaler.pkl"),
    "RNN - 60 hari - 150 layer - 50 epoch": ("RNN_60_150_50.keras", "RNN_60_150_50_scaler.pkl"),
    "RNN - 60 hari - 150 layer - 100 epoch": ("RNN_60_150_100.keras", "RNN_60_150_100_scaler.pkl"),
    "ARIMA":("arima.joblib", "arima_scaler.joblib")
}

def predict_future_df(df_input, target_column, model_path, scaler_path, num_prediction_days=7):
    model_path= f"DATA/{model_path}"
    scaler_path= f"DATA/{scaler_path}"
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    time_steps = model.input_shape[1]
    input_data = df_input[target_column].values
    if len(input_data) < time_steps:
        raise ValueError(f"Data input tidak cukup. Butuh minimal {time_steps} data.")
    last_sequence = np.array(input_data[-time_steps:]).reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)
    input_sequence = last_sequence_scaled.reshape(1, time_steps, 1)
    future_predictions_scaled = []
    for _ in range(num_prediction_days):
        next_pred_scaled = model.predict(input_sequence, verbose=0)
        future_predictions_scaled.append(next_pred_scaled[0, 0])
        input_sequence = np.append(input_sequence[:, 1:, :], [[[next_pred_scaled[0, 0]]]], axis=1)
    predictions_unscaled = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    last_date = df_input.index[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_prediction_days)
    df_history = df_input[[target_column]].copy()
    df_future = pd.DataFrame({target_column: predictions_unscaled.flatten()}, index=prediction_dates)
    df_prediksi = pd.concat([df_history, df_future])
    return df_prediksi

def prediksi_dan_visualisasi(nama_file_model, jumlah_hari, data_historis):
    loaded_model = joblib.load(nama_file_model)
    forecast_values = loaded_model.forecast(steps=jumlah_hari)
    last_date = data_historis.index[-1]
    future_date_index = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                      periods=jumlah_hari, 
                                      freq='D')
    future_forecast = pd.Series(forecast_values.values, index=future_date_index)
    df_prediksi = future_forecast.to_frame(name='Close')
    df_full['is_prediksi'] = False
    df_prediksi['is_prediksi'] = True
    df_gabungan = pd.concat([df_full, df_prediksi])
    df_gabungan = df_gabungan["Close"]

    return df_gabungan




if st.session_state.page == "Menu Utama":
    st.markdown("<h1 style='color:white;'>Dashboard Prediksi Saham BBRI</h1>", unsafe_allow_html=True)
    st.markdown("Visualisasi harga penutupan dan prediksi saham untuk masa depan.")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🔧 Metode Prediksi")
        metode = st.selectbox("Pilih Model", list(MODEL_LIST.keys()), label_visibility="collapsed")
        num_days = st.slider("Pilih jumlah hari prediksi:", min_value=1, max_value=300, value=30, step=1)
        
        if st.button("✅ Jalankan Prediksi"):
            st.session_state.prediksi_dijalankan = True
            st.session_state.metode_terpilih = metode
            
            with st.spinner(f"Menjalankan model {metode} untuk {num_days} hari... Mohon tunggu."):
                if metode == "ARIMA":
                    df_hasil_prediksi = prediksi_dan_visualisasi(
                        nama_file_model='DATA/arima.joblib', 
                        jumlah_hari=num_days, 
                        data_historis=ts_full)

                else:
                    path_model, path_scaler = MODEL_LIST[metode]
                    df_hasil_prediksi = predict_future_df(
                        df_input=df, target_column='Close', model_path=path_model,
                        scaler_path=path_scaler, num_prediction_days=num_days)
            
            st.session_state.df_hasil_prediksi = df_hasil_prediksi

    with col2:
        st.subheader("📊 Info")
        st.info("Halaman ini untuk memprediksi harga di masa depan.")

    st.subheader("Grafik Harga Saham BBRI")
    rentang = st.radio("Tampilkan data untuk:", ["1 Minggu", "1 Bulan", "3 Bulan", "1 Tahun", "3 Tahun", "Semua"], horizontal=True)
    
    tgl_terakhir = df.index.max()
    if rentang == "1 Minggu": tgl_awal = tgl_terakhir - pd.DateOffset(weeks=1)
    elif rentang == "1 Bulan": tgl_awal = tgl_terakhir - pd.DateOffset(months=1)
    elif rentang == "3 Bulan": tgl_awal = tgl_terakhir - pd.DateOffset(months=3)
    elif rentang == "1 Tahun": tgl_awal = tgl_terakhir - pd.DateOffset(years=1)
    elif rentang == "3 Tahun": tgl_awal = tgl_terakhir - pd.DateOffset(years=3)
    else: tgl_awal = df.index.min()
    subset_df = df[df.index >= tgl_awal]
    
    fig = px.line(subset_df, y="Close", title=f"Harga Saham BBRI - {rentang} Terakhir")
    st.plotly_chart(fig, use_container_width=True)

        

    if st.session_state.prediksi_dijalankan and st.session_state.df_hasil_prediksi is not None:
        st.subheader(f"📉 Grafik Prediksi Saham ({st.session_state.metode_terpilih})")
        
        df_hasil = st.session_state.df_hasil_prediksi
        metode = st.session_state.metode_terpilih
        tgl_terakhir_historis = df.index.max()
        tgl_awal_plot = tgl_terakhir_historis - pd.DateOffset(months=3)

        if metode == "ARIMA":
            subset_plot = df_hasil[df_hasil.index >= tgl_awal_plot]
            data_historis = subset_plot[subset_plot.index <= tgl_terakhir_historis]
            data_prediksi = subset_plot[subset_plot.index > tgl_terakhir_historis]
            fig_pred = px.line(title="Harga Aktual vs. Prediksi Masa Depan (ARIMA)")
            fig_pred.add_scatter(x=data_historis.index, y=data_historis.values, name='Harga Aktual',
                                line=dict(color='#1f77b4'))
            fig_pred.add_scatter(x=data_prediksi.index, y=data_prediksi.values, name='Prediksi',
                                line=dict(color='#ff7f0e'))

            st.plotly_chart(fig_pred, use_container_width=True)


        else:
            subset_plot = df_hasil[df_hasil.index >= tgl_awal_plot]
            data_historis = subset_plot[subset_plot.index <= tgl_terakhir_historis]
            data_prediksi = subset_plot[subset_plot.index > tgl_terakhir_historis]
            last_history_point = data_historis.tail(1)
            data_prediksi_nyambung = pd.concat([last_history_point, data_prediksi])
            fig_pred = px.line(data_historis, y="Close", title="Harga Aktual vs. Prediksi Masa Depan")
            fig_pred.data[0].name = 'Harga Aktual'
            fig_pred.data[0].line.color = '#1f77b4'
            fig_pred.add_scatter(
                x=data_prediksi_nyambung.index, 
                y=data_prediksi_nyambung['Close'], 
                mode='lines', 
                name='Prediksi', 
                line=dict(color='#ff7f0e')
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)


# HALAMAN ABOUT
elif st.session_state.page == "About":
    st.markdown("<h1 style='color:white;'>ℹ️ Tentang Aplikasi</h1>", unsafe_allow_html=True)
    st.markdown("""<div style='color:white; font-size:18px;'>Aplikasi ini adalah dashboard...</div>""", unsafe_allow_html=True)


# HALAMAN EVALUASI (DEFAULT)
else:
    st.markdown("<h1 style='color:white;'>Halaman Evaluasi Model</h1>", unsafe_allow_html=True)
    st.markdown("Berikut ini merupakan hasil dari evaluasi yang kami lakukan")
    

    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.subheader("Pilih Model")
        eval_metode = st.selectbox("Pilih model untuk dievaluasi", list(MODEL_LIST.keys()), label_visibility="collapsed")
 
        if st.button("✅ Tampilkan Evaluasi"):
            st.session_state.evaluasi_dijalankan = True
            
            model_file, _= MODEL_LIST[eval_metode]
            # Mengatasi .keras dan .joblib
            model_filename_base = model_file.replace('.keras', '').replace('.joblib', '')

            # Cek file JSON
            json_file_path = f"DATA/eval_{model_filename_base}.json"
            if os.path.exists(json_file_path):
                st.session_state.eval_file_found = True
                with open(json_file_path, 'r') as f:
                    st.session_state.metrik_evaluasi = json.load(f)
            else:
                st.session_state.eval_file_found = False
                st.session_state.metrik_evaluasi = None
            
            # Cek file CSV
            csv_file_path = f"DATA/{model_filename_base}.csv"
            if os.path.exists(csv_file_path):
                st.session_state.csv_file_found = True
                st.session_state.df_eval_chart = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)
            else:
                st.session_state.csv_file_found = False
                st.session_state.df_eval_chart = None


    with col2:
        st.subheader("Akurasi dari File")
        if st.session_state.get('evaluasi_dijalankan', False):
            if st.session_state.get('eval_file_found', False):
                sub_col_rmse, sub_col_mape = st.columns(2, vertical_alignment="top")
                metrics = st.session_state.metrik_evaluasi
                
                with sub_col_rmse:
                    rmse_value = metrics.get("RMSE", "N/A")
                    if isinstance(rmse_value, (int, float)):
                        st.metric(label="RMSE", value=f"{rmse_value:,.4f}")
                    else:
                        st.metric(label="RMSE", value=rmse_value)
                
                with sub_col_mape:
                    mape_value = metrics.get("MAPE", "N/A")
                    if isinstance(mape_value, (int, float)):    
                        filled_value = 100 - mape_value
                        empty_value = mape_value
                        df_donut = pd.DataFrame({'Kategori': ['Terisi', 'Kosong'], 'Nilai': [filled_value, empty_value]})
                        color_filled, color_empty = '#17becf', '#0D1B2A'
                        fig_donut = px.pie(df_donut, values='Nilai', names='Kategori', hole=0.7, color_discrete_map={'Terisi': color_filled, 'Kosong': color_empty})
                        fig_donut.update_traces(textinfo='none', hoverinfo='none', sort=False, marker=dict(line=dict(color=color_empty, width=2)))
                        center_text = f"MAPE<br><b style='font-size:24px; color:{color_filled};'>{mape_value:.2f}%</b>"
                        fig_donut.update_layout(width=180, height=180, showlegend=False, margin=dict(t=0, b=0, l=0, r=0), annotations=[dict(text=center_text, x=0.5, y=0.5, font_size=18, showarrow=False)], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_donut, use_container_width=False, config={'displayModeBar': False})
                    else:
                        st.metric(label="MAPE (%)", value=mape_value)
            else:
                st.error("File metrik tidak ditemukan.")
        else:
            st.info("Pilih model dan klik tombol untuk melihat hasil.")


    # KOLOM 3: PREVIEW GRAFIK KECIL
    with col3:
            st.subheader("Preview Akhir")
            if st.session_state.get('evaluasi_dijalankan', False):
                if st.session_state.get('csv_file_found', False) and st.session_state.df_eval_chart is not None:
                    df_chart = st.session_state.df_eval_chart
                    
                    # Menyesuaikan nama kolom berdasarkan model
                    if eval_metode == "ARIMA":
                        y_cols = ['Aktual', 'Prediksi']
                    else:
                        y_cols = ['Harga_Aktual', 'Prediksi_Uji']

                    preview_len = int(len(df_chart) * 0.25)
                    df_preview = df_chart.tail(preview_len)
                    
                    fig_preview = px.line(df_preview, y=y_cols, color_discrete_map={y_cols[0]: '#1f77b4', y_cols[1]: '#ff7f0e'})
                    fig_preview.update_layout(
                        showlegend=False,
                        margin=dict(t=5, b=5, l=5, r=5),
                        height=200,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor="#1B263B" 
                    )
                    st.plotly_chart(fig_preview, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.error("File grafik tidak ditemukan.")
            else:
                st.info("Grafik preview akan muncul di sini.")

    # TAMPILAN GRAFIK BESAR (DI BAWAH 3 KOLOM)
    st.divider() 
    if st.session_state.get('evaluasi_dijalankan', False):
        if st.session_state.get('csv_file_found', False) and st.session_state.df_eval_chart is not None:
            st.subheader("Grafik Perbandingan Lengkap")
            df_eval_chart = st.session_state.df_eval_chart 
            
            # Menyesuaikan nama kolom dan label berdasarkan model
            if eval_metode == "ARIMA":
                y_cols = ['Aktual', 'Prediksi']
                labels = {'Aktual': 'Harga Aktual', 'Prediksi': 'Prediksi pada Data Uji'}
            else:
                y_cols = ['Harga_Aktual', 'Prediksi_Uji']
                labels = {'Harga_Aktual': 'Harga Aktual', 'Prediksi_Uji': 'Prediksi pada Data Uji'}

            fig_eval = px.line(df_eval_chart, y=y_cols, labels=labels, color_discrete_map={y_cols[0]: '#1f77b4', y_cols[1]: '#ff7f0e'})
            fig_eval.update_layout(
                title=f'Visualisasi Hasil Evaluasi Model',
                xaxis_title='Tanggal', yaxis_title='Harga', legend_title_text='Keterangan'
            )
            st.plotly_chart(fig_eval, use_container_width=True)
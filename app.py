import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import tensorflow as tf
from PIL import Image
import json

# Setup halaman utama
st.set_page_config(page_title="NutriCheck AI", page_icon="🥗", layout="wide")

# Load Semua Aset AI Lokal (Tanpa Internet / API luar)
@st.cache_resource
def load_lokal_ai():
    model_medis = joblib.load('diabetes_rf_model.pkl')
    model_vision = tf.keras.models.load_model('model_makanan_lokal.h5')
    with open('label_makanan.json', 'r') as f:
        labels = json.load(f)
    return model_medis, model_vision, labels

try:
    ai_diabetes, ai_vision, daftar_label = load_lokal_ai()
except Exception as e:
    st.error(f"Gagal memuat AI. Pastikan kedua Notebook sudah dijalankan! Eror: {e}")
    st.stop()

# ==============================================================================
# KAMUS DATABASE GIZI LOKAL (SUDAH DISINKRONKAN DENGAN NAMA FOLDER KAGGLE KAMU)
# ==============================================================================
database_gizi = {
    '01_nasi_goreng': {'nama': 'Nasi Goreng Ayam', 'kalori': 650, 'gula': 8, 'natrium': 1200, 'lemak': 22},
    '02_mie_goreng': {'nama': 'Mie Instan Goreng', 'kalori': 380, 'gula': 7, 'natrium': 1070, 'lemak': 14},
    '03_gado_gado': {'nama': 'Gado-Gado', 'kalori': 350, 'gula': 12, 'natrium': 600, 'lemak': 16},
    '04_ayam_goreng': {'nama': 'Ayam Goreng Fastfood', 'kalori': 290, 'gula': 1, 'natrium': 610, 'lemak': 18},
    '05_bubur_ayam': {'nama': 'Bubur Ayam', 'kalori': 370, 'gula': 3, 'natrium': 800, 'lemak': 12},
    '06_salad_sayur': {'nama': 'Salad Sayur', 'kalori': 150, 'gula': 4, 'natrium': 200, 'lemak': 7},
    '07_martabak_manis': {'nama': 'Martabak Manis', 'kalori': 620, 'gula': 45, 'natrium': 450, 'lemak': 28},
    '08_nasi_padang': {'nama': 'Nasi Padang (Rendang + Tunjang)', 'kalori': 850, 'gula': 6, 'natrium': 1450, 'lemak': 42},
    '09_bakso_sapi': {'nama': 'Bakso Sapi Kuah Komplit', 'kalori': 520, 'gula': 4, 'natrium': 1620, 'lemak': 26},
    '10_sate_kambing': {'nama': 'Sate Kambing (10 Tusuk)', 'kalori': 610, 'gula': 15, 'natrium': 1100, 'lemak': 32},
    '11_soto_ayam': {'nama': 'Soto Ayam Lamongan', 'kalori': 420, 'gula': 3, 'natrium': 1350, 'lemak': 18},
    '12_ayam_geprek': {'nama': 'Ayam Geprek Sambal Bawang', 'kalori': 680, 'gula': 2, 'natrium': 1150, 'lemak': 38},
    '13_gorengan': {'nama': 'Gorengan Campur (3 Biji)', 'kalori': 450, 'gula': 2, 'natrium': 680, 'lemak': 25},
    '14_cireng': {'nama': 'Cireng Bumbu Rujak (5 Biji)', 'kalori': 350, 'gula': 5, 'natrium': 720, 'lemak': 15},
    '15_siomay': {'nama': 'Siomay Bandung (1 Porsi)', 'kalori': 480, 'gula': 8, 'natrium': 950, 'lemak': 18},
    '16_es_kopi_susu': {'nama': 'Es Kopi Susu Gula Aren', 'kalori': 250, 'gula': 28, 'natrium': 120, 'lemak': 9},
    '17_boba': {'nama': 'Bobba Milk Tea', 'kalori': 380, 'gula': 38, 'natrium': 150, 'lemak': 12},
    '18_es_cendol': {'nama': 'Es Cendol', 'kalori': 290, 'gula': 30, 'natrium': 180, 'lemak': 8}
}

st.title("🥗 NutriCheck AI — Full Local Embedded Intelligent System")
st.write("Sistem terintegrasi Computer Vision & Predictive Analytics yang berjalan 100% lokal di komputer Anda.")
st.markdown("---")

# --- SIDEBAR PROFIL USER ---
st.sidebar.header("📋 Profil Fisik Anda")
pilihan_tensi = st.sidebar.selectbox("Apakah Tensi Anda Sering Tinggi?", ["Tidak", "Ya"])
pilihan_kolesterol = st.sidebar.selectbox("Apakah Kolesterol Anda Tinggi?", ["Tidak", "Ya"])
input_bmi = st.sidebar.slider("Indeks Massa Tubuh (BMI)", 15, 45, 24)
pilihan_rokok = st.sidebar.selectbox("Apakah Anda Merokok?", ["Tidak", "Ya"])

kategori_usia = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+"]
pilihan_usia = st.sidebar.selectbox("Kelompok Usia Anda", kategori_usia)

tensi_num = 1 if pilihan_tensi == "Ya" else 0
kol_num = 1 if pilihan_kolesterol == "Ya" else 0
rokok_num = 1 if pilihan_rokok == "Ya" else 0
usia_num = kategori_usia.index(pilihan_usia) + 1

# --- MULTI-TAB DASHBOARD ---
tab1, tab2 = st.tabs(["📸 Scan Foto (AI Vision Lokal)", "🧬 Prediksi Risiko Medis"])

with tab1:
    st.subheader("Pindai Menggunakan Kamera / Galeri Perangkat")
    col_in, col_out = st.columns(2)
    img_ready = None
    
    with col_in:
        sumber_foto = st.radio("Metode Input Gambar:", ["Gunakan Kamera Perangkat", "Unggah dari Galeri"])
        if sumber_foto == "Gunakan Kamera Perangkat":
            foto_file = st.camera_input("Ambil foto hidangan")
            if foto_file: img_ready = Image.open(foto_file)
        else:
            foto_file = st.file_uploader("Pilih berkas gambar", type=["jpg", "png", "jpeg"])
            if foto_file:
                img_ready = Image.open(foto_file)
                st.image(img_ready, width=280)

    with col_out:
        if img_ready:
            # PROSES PREDIKSI GAMBAR OLEH CNN LOKAL KAMU
            img_resized = img_ready.resize((128, 128))
            img_array = np.array(img_resized) / 255.0  # Normalisasi warna
            img_array = np.expand_dims(img_array, axis=0)  # Bentuk tensor batch
            
            preds = ai_vision.predict(img_array)[0]
            indeks_tertinggi = np.argmax(preds)
            folder_terdeteksi = daftar_label[indeks_tertinggi]
            skor_pasti = preds[indeks_tertinggi] * 100
            
            # Format tampilan nama agar angka folder di depan hilang saat dicetak ke user
            nama_tampilan = folder_terdeteksi.split('_', 1)[-1].replace('_', ' ').upper()
            st.success(f"### 🤖 AI Mendeteksi: **{nama_tampilan}** ({skor_pasti:.1f}% Akurat)")
            
            # Ambil data gizi berdasarkan tebakan folder model CNN asli
            if folder_terdeteksi in database_gizi:
                gizi = database_gizi[folder_terdeteksi]
                
                st.info(f"• **Nama Menu:** {gizi['nama']}\n\n"
                        f"• 🔥 **Kalori:** {gizi['kalori']} kkal\n\n"
                        f"• 🍬 **Gula:** {gizi['gula']} gram\n\n"
                        f"• 🧂 **Garam:** {gizi['natrium']} mg\n\n"
                        f"• 🥩 **Lemak:** {gizi['lemak']} gram")
                
                # Grafik Batang Plotly
                df_grafik = pd.DataFrame({
                    'Zat Gizi': ['Gula (g)', 'Garam (mg/10)', 'Lemak (g)'],
                    'Jumlah': [gizi['gula'], gizi['natrium']/10, gizi['lemak']]
                })
                fig = px.bar(df_grafik, x='Zat Gizi', y='Jumlah', color='Zat Gizi', height=240, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig, use_container_width=True)
                
                # Evaluasi Rekomendasi Personal Sesuai Profil User
                st.markdown("### 💡 Evaluasi Profil Gizi:")
                pemicu_warning = False
                if gizi['gula'] >= 25:
                    st.error("🚨 Makanan/Minuman ini mengandung gula yang sangat tinggi. Kurangi porsi konsumsi Anda.")
                    pemicu_warning = True
                if gizi['natrium'] >= 1100 and tensi_num == 1:
                    st.warning("⚠️ Menu ini sangat asin sedangkan Anda memiliki riwayat tensi tinggi. Disarankan batasi kuah/bumbu.")
                    pemicu_warning = True
                if gizi['lemak'] >= 25 and kol_num == 1:
                    st.warning("⚠️ Kadar lemak pada hidangan ini cukup tinggi untuk kondisi riwayat kolesterol Anda.")
                    pemicu_warning = True
                if not pemicu_warning:
                    st.success("✅ Kandungan menu aman dan proporsional untuk dikonsumsi dengan profil fisik Anda hari ini.")
            else:
                st.warning(f"Menu '{folder_terdeteksi}' terdeteksi namun belum didaftarkan di kamus database_gizi lokal Anda.")
        else:
            st.info("Masukkan foto hidangan untuk memicu kalkulasi neural network lokal.")

with tab2:
    st.subheader("🔮 Kalkulasi Probabilitas Risiko Diabetes (Random Forest)")
    data_pasien = np.array([[tensi_num, kol_num, input_bmi, rokok_num, usia_num]])
    persentase_risiko = ai_diabetes.predict_proba(data_pasien)[0][1] * 100
    
    st.metric(label="Skor Risiko Diabetes Anda", value=f"{persentase_risiko:.1f} %")
    if persentase_risiko < 30:
        st.success("Kategori: RISIKO RENDAH")
    elif 30 <= persentase_risiko < 60:
        st.warning("Kategori: RISIKO SEDANG")
    else:
        st.error("Kategori: RISIKO TINGGI")
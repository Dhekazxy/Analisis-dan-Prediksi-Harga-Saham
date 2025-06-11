# 📊 Prediksi Harga Saham BBRI (2015–2025)

Proyek ini bertujuan untuk memprediksi harga saham **PT Bank Rakyat Indonesia Tbk (BBRI)** menggunakan tiga metode: **ARIMA**, **RNN**, dan **LSTM**. Data diambil dari Yahoo Finance periode **2015 hingga Mei 2025**.

---

## ✅ Hasil Evaluasi Model
| Model  | Konfigurasi Terbaik                    | MAPE (%) |
|--------|----------------------------------------|----------|
| LSTM   | 60 hari, 4 layer, 100 epoch            | **1.53** |
| RNN    | 30 hari, 2 layer (50 & 100), 100 epoch | 1.70     |
| ARIMA  | Auto ARIMA (order = 0, 1, 0)           | 12.96    |

---

## 🧾 Kesimpulan

Model **LSTM** menunjukkan performa terbaik dalam memprediksi harga saham BBRI, diikuti oleh **RNN**.  
Model **ARIMA** memiliki akurasi terendah karena keterbatasannya dalam mengenali pola non-linear.  
Hasil ini menunjukkan bahwa model berbasis deep learning lebih unggul dalam prediksi saham jangka panjang.

---

## 👥 Anggota Kelompok 4
- Dimas Kurniawan – 23031554067  
- Eko Hadi Prasetiyo – 23031554121  
- M. Hilmi Musyaffa – 23031554128

---


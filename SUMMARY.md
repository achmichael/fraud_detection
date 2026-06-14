# SUMMARY — Credit Card Fraud Detection (XGBoost)

## 1) Gambaran Umum Proyek
Proyek ini membangun sistem klasifikasi untuk mendeteksi transaksi kartu kredit yang berpotensi fraud menggunakan **XGBoost** pada dataset tabular yang sangat tidak seimbang (imbalanced).

Tujuan utama:
- Mendeteksi sebanyak mungkin transaksi fraud (recall tinggi)
- Tetap menjaga kualitas alert (precision memadai)
- Menghindari ketergantungan pada metrik accuracy yang menyesatkan pada data imbalance ekstrem

Model utama:
- **XGBoost Classifier** dengan penanganan imbalance melalui `scale_pos_weight`

Baseline pembanding:
- **Logistic Regression** dengan `class_weight="balanced"`

---

## 2) Struktur File Proyek
- `credit_card_fraud_xgboost_project.py` → script utama end-to-end (EDA, preprocessing, training, evaluasi, tuning threshold, baseline, visualisasi)
- `credit_card_fraud_xgboost_project.ipynb` → versi notebook
- `hasil_project.md` → ringkasan hasil eksperimen
- `requirements.txt` → daftar dependensi
- `SUMMARY.md` → ringkasan proyek (file ini)

Dataset yang digunakan:
- `data/creditcard.csv` (atau `/content/creditcard.csv` saat dijalankan di Google Colab)

---

## 3) Dependensi
Isi `requirements.txt`:
- pandas
- numpy==1.26.4
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- shap==0.45.1

Catatan:
- `shap` dipakai opsional untuk interpretasi model. Jika environment tidak kompatibel, pipeline utama tetap bisa berjalan.

---

## 4) Ringkasan Dataset
- Ukuran data: **284.807 baris, 31 kolom**
- Target: `Class`
  - `0` = normal
  - `1` = fraud
- Missing value: **0**
- Duplikasi: **1.081 baris**

Distribusi kelas:
- Normal: **284.315 (99,8273%)**
- Fraud: **492 (0,1727%)**
- Rasio normal:fraud ≈ **577,88 : 1**

Implikasi:
- Accuracy tidak layak jadi metrik utama
- Fokus evaluasi ke: **precision, recall, F1, ROC-AUC, PR-AUC**

---

## 5) Alur Pengerjaan (Pipeline)

### a. Data Loading & Validasi Awal
- Load CSV
- Cek bentuk data, nama kolom, tipe data
- Cek missing value dan duplikasi

### b. Exploratory Data Analysis (EDA)
- Analisis distribusi kelas
- Analisis statistik `Amount` dan `Time` per kelas
- Visualisasi distribusi `Amount` dan `Time`
- Korelasi fitur terhadap target `Class`

### c. Preprocessing
- `X` = semua kolom selain `Class`
- `y` = kolom `Class`
- Split data stratified:
  - Train/Test = 80/20
  - Train dipecah lagi jadi Train/Validation untuk threshold tuning
- Scaling dengan `RobustScaler` hanya untuk:
  - `Time`
  - `Amount`
- Fitur `V1`–`V28` tidak di-scale ulang (sudah fitur PCA)

### d. Penanganan Imbalance
- Hitung `scale_pos_weight` dari data training:
  - Normal training: 170.588
  - Fraud training: 295
  - `scale_pos_weight`: **578,2644**

### e. Modeling
- Model utama: `XGBClassifier`
  - `objective="binary:logistic"`
  - `eval_metric="aucpr"`
  - parameter utama: n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.90, colsample_bytree=0.90, tree_method="hist"

### f. Evaluasi
- Prediksi probabilitas
- Evaluasi di threshold default `0.5`
- Metrik:
  - Classification report
  - Confusion matrix
  - ROC-AUC
  - PR-AUC

### g. Threshold Tuning
- Uji threshold: 0.1 s.d. 0.9 pada validation set
- Pilih threshold terbaik berdasarkan **F2-score** (lebih menekankan recall)
- Evaluasi akhir threshold terpilih di test set

### h. Interpretasi Model
- Feature importance XGBoost (Top 15)
- SHAP summary plot (opsional)

### i. Baseline Comparison
- Bandingkan XGBoost vs Logistic Regression balanced

---

## 6) Hasil Utama

### XGBoost — Test set, threshold 0.5
- Precision fraud: **0,6774**
- Recall fraud: **0,8571**
- F1 fraud: **0,7568**
- ROC-AUC: **0,9759**
- PR-AUC: **0,8541**

Confusion matrix (0.5):
- TN = 56.824
- FP = 40
- FN = 14
- TP = 84

### Threshold tuning
Threshold terpilih dari validation: **0,9** (berdasarkan F2 tertinggi di validation).

### XGBoost — Test set, threshold 0.9
- Precision fraud: **0,8333**
- Recall fraud: **0,8163**
- F1 fraud: **0,8247**
- ROC-AUC: **0,9759**
- PR-AUC: **0,8541**

Confusion matrix (0.9):
- TN = 56.848
- FP = 16
- FN = 18
- TP = 80

Trade-off:
- Threshold 0.5: recall lebih tinggi, tapi FP lebih banyak
- Threshold 0.9: precision lebih tinggi, FP turun, tapi fraud terlewat (FN) naik

---

## 7) Perbandingan dengan Baseline
| Model | Threshold | Precision fraud | Recall fraud | F1 fraud | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression balanced | 0,5 | 0,0616 | 0,9184 | 0,1154 | 0,7209 |
| XGBoost default | 0,5 | 0,6774 | 0,8571 | 0,7568 | 0,8541 |
| XGBoost selected | 0,9 | 0,8333 | 0,8163 | 0,8247 | 0,8541 |

Kesimpulan komparatif:
- Logistic Regression memberi recall tinggi, tapi precision sangat rendah (alert palsu sangat banyak)
- XGBoost memberi keseimbangan jauh lebih baik untuk fraud detection pada data ini

---

## 8) Feature Importance (XGBoost)
Top fitur penting:
1. V14
2. V10
3. V4
4. Amount
5. V8
6. V20
7. V12
8. V13
9. V11
10. V19
11. V17
12. V21
13. V3
14. V26
15. V5

Catatan:
- Karena fitur V1–V28 hasil PCA dan anonim, interpretasi bisnis langsung terbatas
- Ranking tetap berguna untuk melihat sinyal numerik dominan yang dipakai model

---

## 9) Cara Menjalankan Proyek

### Opsi 1 — Script Python
1. Pastikan dataset ada di `data/creditcard.csv`
2. Install dependensi:
```bash
pip install -r requirements.txt
```
3. Jalankan script:
```bash
python credit_card_fraud_xgboost_project.py
```

### Opsi 2 — Notebook
1. Buka `credit_card_fraud_xgboost_project.ipynb`
2. Jalankan cell berurutan dari atas ke bawah
3. Jika di Colab, upload `creditcard.csv` atau sesuaikan path

---

## 10) Keputusan Operasional yang Disarankan
- Jika prioritas bisnis: **tangkap fraud semaksimal mungkin**, gunakan threshold lebih rendah (contoh 0.5)
- Jika prioritas bisnis: **kurangi false alarm/manual review**, gunakan threshold lebih tinggi (contoh 0.9)
- Threshold final harus ditetapkan berdasarkan:
  - biaya false negative (fraud lolos)
  - biaya false positive (investigasi sia-sia)
  - kapasitas tim investigasi

---

## 11) Keterbatasan Saat Ini
- Threshold dipilih dari satu split validation; belum cross-validation menyeluruh
- Belum ada optimasi hiperparameter ekstensif
- Belum eksplisit memasukkan fungsi biaya bisnis (cost-sensitive objective)
- Belum ada simulasi drift/monitoring untuk deployment jangka panjang

---

## 12) Rekomendasi Pengembangan Lanjutan
1. Hyperparameter tuning XGBoost dengan Stratified K-Fold
2. Optimasi threshold berbasis cost matrix bisnis
3. Uji strategi sampling (mis. SMOTE) hanya di training, lalu bandingkan PR-AUC/recall
4. Tambahkan monitoring data drift dan model decay
5. Perluas interpretabilitas dengan SHAP global + local explanation

---

## 13) Kesimpulan Akhir
Proyek berhasil membangun pipeline fraud detection yang kuat untuk data sangat imbalanced. **XGBoost** menunjukkan performa unggul dibanding baseline linear, dengan **PR-AUC tinggi** dan trade-off precision-recall yang bisa diatur lewat threshold. Sistem siap menjadi fondasi untuk tahap lanjutan: tuning berbasis biaya bisnis dan persiapan deployment production.

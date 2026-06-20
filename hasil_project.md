# Dokumentasi & Hasil Running: Credit Card Fraud Detection dengan XGBoost

Dokumen ini berisi hasil eksekusi (*running*) terbaru dari *notebook* `credit_card_fraud_xgboost_project.ipynb` beserta penjelasan detail untuk setiap tahapan (cell) yang dieksekusi. Dokumen ini sangat cocok digunakan sebagai panduan presentasi atau demo proyek.

## 1. Load dan Pahami Dataset (Cell 1 - 9)
**Penjelasan Cell:**
Tahap awal mengimpor *library* yang dibutuhkan (Pandas, Scikit-Learn, XGBoost) dan memuat dataset `creditcard.csv`. Selanjutnya, dilakukan pengecekan ukuran data dan distribusi kelas target (`Class`).

**Hasil Output:**
```text
Ukuran dataset: (284807, 31)
Jumlah transaksi normal: 284,315
Jumlah transaksi fraud : 492
Persentase normal      : 99.8273%
Persentase fraud       : 0.1727%
Rasio normal:fraud     : 577.88:1
```

**Interpretasi untuk Demo:**
Kasus *fraud* hanya sebagian sangat kecil dari dataset (kurang dari 0.2%). Ini disebut *Extreme Class Imbalance*. Model konvensional yang hanya menebak "Normal" akan mendapatkan akurasi 99.8%, namun gagal mendeteksi penipuan. Karenanya, metrik *Accuracy* tidak relevan di sini.

---

## 2. Exploratory Data Analysis / EDA (Cell 10 - 19)
**Penjelasan Cell:**
Melakukan pengecekan *missing value*, duplikasi, dan menganalisis statistik dari fitur `Amount` (nominal transaksi) dan `Time` (waktu transaksi). Selain itu, dihitung juga korelasi fitur terhadap target.

**Hasil Output:**
```text
Total missing value: 0
Jumlah baris duplikat: 1081

Statistik Amount per kelas:
           mean         std  min   25%    50%     75%       max
Class                                                          
0      88.291022  250.105092  0.0  5.65  22.00   77.05  25691.16
1     122.211321  256.683288  0.0  1.00   9.25  105.89   2125.87

Top 5 Korelasi Absolut Terhadap Target:
V17: -0.326481
V14: -0.302544
V12: -0.260593
V10: -0.216883
V16: -0.196539
```

**Interpretasi untuk Demo:**
Nilai median `Amount` pada transaksi *fraud* ternyata lebih kecil (9.25) dibandingkan transaksi normal (22.00). Ini membuktikan penipu sering melakukan transaksi nominal kecil agar tidak mencurigakan. Selain itu, korelasi fitur sangat rendah (maksimal ~0.3), menandakan perlunya model yang bisa menangkap interaksi *non-linear* seperti XGBoost.

---

## 3. Preprocessing & Splitting (Cell 20 - 22)
**Penjelasan Cell:**
Membagi data menjadi set *Train* (60%), *Validation* (20%), dan *Test* (20%) secara *Stratified* (mempertahankan rasio *fraud*). Fitur `Time` dan `Amount` disesuaikan skalanya dengan `RobustScaler` karena rentan terhadap nilai *outlier* (pencilan).

**Hasil Output:**
```text
Ukuran train     : (170883, 30)
Ukuran validation: (56962, 30)
Ukuran test      : (56962, 30)

Distribusi kelas train:
0    170588
1       295
```

**Interpretasi untuk Demo:**
*Stratified split* sukses mempertahankan proporsi kelas fraud. Dari 170 ribu data training, model hanya akan belajar dari 295 contoh kasus *fraud*.

---

## 4. Handling Imbalanced Data (Cell 23 - 25)
**Penjelasan Cell:**
Menghitung `scale_pos_weight` untuk disuntikkan ke parameter algoritma XGBoost. Ini berfungsi agar model memberikan perhatian ekstra saat mempelajari kasus *fraud*.

**Hasil Output:**
```text
Normal pada training: 170,588
Fraud pada training : 295
scale_pos_weight    : 578.2644
```

**Interpretasi untuk Demo:**
Angka ini menginstruksikan XGBoost: *"Satu kesalahan prediksi pada kelas fraud setara dengan 578 kesalahan prediksi pada kelas normal!"* Hal ini mencegah model hanya condong ke mayoritas kelas normal.

---

## 5. Modeling Baseline XGBoost (Cell 28 - 30)
**Penjelasan Cell:**
Melatih model XGBoost dengan hyperparameter untuk mencegah *overfitting* (seperti `max_depth=4`, `learning_rate=0.05`). Setelah dilatih, model dievaluasi di *test set* dengan `threshold` bawaan 0.5.

**Hasil Output:**
```text
Model: XGBoost default threshold (0.5)

Confusion matrix:
[[56826    38]
 [   14    84]]

Classification report:
              precision    recall  f1-score   support
           0     0.9998    0.9993    0.9995     56864
           1     0.6885    0.8571    0.7636        98

ROC-AUC : 0.9730
PR-AUC  : 0.8481
```

**Interpretasi untuk Demo:**
Dari 98 total penipuan (*fraud*) yang ada, XGBoost berhasil mendeteksi **84 kasus** (`Recall = 0.8571`) dan hanya kecolongan 14 kasus (`False Negative`). Namun, ia juga salah menuduh 38 transaksi normal sebagai penipuan (`False Positive`, memengaruhi `Precision`).

---

## 6. Threshold Tuning (Cell 31 - 34)
**Penjelasan Cell:**
Untuk mencari keseimbangan terbaik antara *Recall* dan *Precision*, kita menguji berbagai batas probabilitas (*threshold*) menggunakan data *Validation*, lalu memilih yang memiliki **F2-Score** tertinggi.

**Hasil Output:**
```text
Threshold terbaik berdasarkan validation F2-score:
threshold            0.7
precision_fraud      0.8041
recall_fraud         0.7878
f1_fraud             0.7959
f2_fraud             0.7910

Evaluasi XGBoost dengan Threshold 0.7 pada Test Set:
Confusion matrix:
[[56837    27]
 [   16    82]]

              precision    recall  f1-score
           1     0.7523    0.8367    0.7923
```

**Interpretasi untuk Demo:**
Tuning mengubah *threshold* menjadi 0.7. Pada uji coba di data *Test*, *Recall* sedikit turun dari 84 menjadi 82 kasus yang tertangkap. Namun sebagai gantinya, salah tangkap (*False Positive*) berhasil ditekan dari 38 menjadi hanya 27. Dalam dunia perbankan, pemilihan *threshold* (apakah 0.5 atau 0.7) bergantung sepenuhnya pada limitasi kapasitas tim investigator kartu kredit.

---

## 7. Feature Importance & Interpretability (Cell 35 - 39)
**Penjelasan Cell:**
Menampilkan fitur apa saja yang paling berperan penting bagi model XGBoost dalam mendeteksi *fraud*. *(Catatan: Pada iterasi run lingkungan lokal saat ini, modul SHAP dilewati karena benturan versi string dependensi)*.

**Hasil Output:**
```text
Top 5 Feature Importance:
14     V14    0.376476
12     V12    0.074034
10     V10    0.070451
4       V4    0.057825
20     V20    0.037342
```

**Interpretasi untuk Demo:**
Fitur `V14`, `V12`, dan `V10` paling dominan sebagai acuan XGBoost (menyumbang hampir 50% keputusan). Karena ini adalah hasil PCA yang dianonimkan untuk menjaga kerahasiaan nasabah bank, kita tidak dapat menafsirkan fitur aslinya. Akan tetapi, representasi ini penting untuk menunjukkan bahwa model *Machine Learning* bukanlah *Black Box* total, melainkan punya acuan matematis yang dapat dipertanggungjawabkan logikanya.

---

## 8. Baseline: Logistic Regression (Cell 40 - 43)
**Penjelasan Cell:**
Membandingkan model canggih XGBoost dengan model linier klasik, yaitu *Logistic Regression* bersistem klasifikasi berimbang (`class_weight="balanced"`).

**Hasil Output:**
```text
                           model  threshold  precision_fraud  recall_fraud  f1_fraud    pr_auc  
1      XGBoost default threshold        0.5         0.688525      0.857143  0.763636  0.848096  
2     XGBoost selected threshold        0.7         0.752294      0.836735  0.792271  0.848096  
0   Logistic Regression balanced        0.5         0.061560      0.918367  0.115385  0.720881  

Confusion matrix Logistic Regression:
[[55492  1372]
 [    8    90]]
```

**Interpretasi untuk Demo:**
*Logistic Regression* mendeteksi lebih banyak penipu (90 dari 98, *Recall* sangat tinggi). Namun model linier konvensional ini **menuduh 1.372 transaksi normal sebagai penipuan!** (`Precision` hancur menjadi ~6%). Jika dipakai di dunia nyata, divisi *Customer Service* bank akan kolaps akibat komplain pelanggan sah yang transaksinya ditolak. Itulah pembuktian mengapa kita wajib beralih menggunakan algoritma dengan arsitektur pohon berlapis yang menangkap interaksi *non-linear* seperti XGBoost karena nilai *PR-AUC* nya jauh lebih tinggi (0.84 berbanding 0.72).

---

## 9. Kesimpulan & Saran Pengembangan
XGBoost dengan penyetelan `scale_pos_weight` telah berhasil memberikan performa luar biasa dalam mengatasi tantangan *Extreme Imbalanced Data*.
- **Threshold Default (0.5)**: Ideal diaplikasikan jika institusi menetapkan langkah pencegahan agresif guna menangkap *fraud* semaksimal mungkin.
- **Threshold Disetel (0.7)**: Sangat direkomendasikan bila institusi ingin efisiensi waktu karena dapat meredam alarm *False Positive* tanpa mengorbankan angka penangkapan secara masif.

**Saran Inovasi Lanjutan:**
1. Menerapkan skema evaluasi *Stratified K-Fold Cross Validation* guna memastikan model lebih matang dalam mempelajari variasi data penipuan.
2. Sinkronisasi versi *library* agar proses analisis fitur eksplanatori mendalam (*SHAP values plot*) dapat berjalan lancar.
3. Pengembangan matriks biaya operasional (*cost-sensitive tuning matrix*) agar *threshold* bisa langsung terhubung ke estimasi kerugian dalam nilai mata uang asli.

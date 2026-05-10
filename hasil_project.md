# Hasil Project Credit Card Fraud Detection - XGBoost

## Ringkasan dataset

- File dataset: `data/creditcard.csv`
- Ukuran dataset: 284.807 baris dan 31 kolom
- Target: `Class`
- `Class = 0`: transaksi normal
- `Class = 1`: transaksi fraud
- Missing value: 0
- Duplikasi: 1.081 baris

## Masalah imbalance

Distribusi kelas:

| Class | Jumlah | Persentase |
|---|---:|---:|
| 0 - Normal | 284.315 | 99,8273% |
| 1 - Fraud | 492 | 0,1727% |

Rasio normal terhadap fraud sekitar 577,88:1. Karena itu, accuracy tidak dipakai
sebagai metrik utama. Fokus evaluasi adalah recall, precision, F1-score,
ROC-AUC, dan PR-AUC.

## Alasan menggunakan XGBoost

XGBoost dipilih karena cocok untuk data tabular numerik, mampu menangkap pola
non-linear dari kombinasi fitur, dan menyediakan `scale_pos_weight` untuk
membantu model belajar dari kelas fraud yang sangat sedikit.

## Preprocessing

- Fitur `X`: semua kolom selain `Class`.
- Target `y`: kolom `Class`.
- `Time` dan `Amount` di-scale menggunakan `RobustScaler`.
- `V1` sampai `V28` tidak di-scale ulang karena merupakan fitur hasil PCA.
- Split data dilakukan secara stratified.
- Test set 20% disimpan sebagai evaluasi akhir.
- Dari training set, sebagian data dipakai sebagai validation set untuk
  threshold tuning agar pemilihan threshold tidak memakai test set.
- `scale_pos_weight` dihitung dari training data:
  - Normal training: 170.588
  - Fraud training: 295
  - `scale_pos_weight`: 578,2644

## Hasil evaluasi XGBoost

Pada test set dengan threshold default 0,5:

| Metrik fraud | Nilai |
|---|---:|
| Precision | 0,6774 |
| Recall | 0,8571 |
| F1-score | 0,7568 |
| ROC-AUC | 0,9759 |
| PR-AUC | 0,8541 |

Confusion matrix threshold 0,5:

| | Pred normal | Pred fraud |
|---|---:|---:|
| Actual normal | 56.824 | 40 |
| Actual fraud | 14 | 84 |

## Threshold tuning

Threshold tuning dilakukan pada validation set. Ringkasan hasil:

| Threshold | Precision fraud | Recall fraud | F1 fraud | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,1 | 0,2828 | 0,8485 | 0,4242 | 56.650 | 213 | 15 | 84 |
| 0,2 | 0,4576 | 0,8182 | 0,5870 | 56.767 | 96 | 18 | 81 |
| 0,3 | 0,6045 | 0,8182 | 0,6953 | 56.810 | 53 | 18 | 81 |
| 0,4 | 0,6810 | 0,7980 | 0,7349 | 56.826 | 37 | 20 | 79 |
| 0,5 | 0,7290 | 0,7879 | 0,7573 | 56.834 | 29 | 21 | 78 |
| 0,6 | 0,7573 | 0,7879 | 0,7723 | 56.838 | 25 | 21 | 78 |
| 0,7 | 0,8041 | 0,7879 | 0,7959 | 56.844 | 19 | 21 | 78 |
| 0,8 | 0,8387 | 0,7879 | 0,8125 | 56.848 | 15 | 21 | 78 |
| 0,9 | 0,8667 | 0,7879 | 0,8254 | 56.851 | 12 | 21 | 78 |

Threshold 0,9 dipilih berdasarkan F2-score validation tertinggi. Pada validation
set, threshold ini mempertahankan recall yang sama dengan beberapa threshold
lebih rendah, tetapi mengurangi false positive.

Pada test set dengan threshold 0,9:

| Metrik fraud | Nilai |
|---|---:|
| Precision | 0,8333 |
| Recall | 0,8163 |
| F1-score | 0,8247 |
| ROC-AUC | 0,9759 |
| PR-AUC | 0,8541 |

Confusion matrix threshold 0,9:

| | Pred normal | Pred fraud |
|---|---:|---:|
| Actual normal | 56.848 | 16 |
| Actual fraud | 18 | 80 |

Trade-off: threshold 0,5 menemukan fraud lebih banyak pada test set (recall
0,8571) tetapi menghasilkan 40 false positive. Threshold 0,9 menghasilkan
precision lebih tinggi dan false positive lebih sedikit, tetapi fraud yang lolos
menjadi 18 kasus.

## Feature importance

Top 15 fitur paling penting dari XGBoost:

| Fitur | Importance |
|---|---:|
| V14 | 0,3363 |
| V10 | 0,1663 |
| V4 | 0,0637 |
| Amount | 0,0412 |
| V8 | 0,0368 |
| V20 | 0,0347 |
| V12 | 0,0307 |
| V13 | 0,0265 |
| V11 | 0,0219 |
| V19 | 0,0211 |
| V17 | 0,0207 |
| V21 | 0,0185 |
| V3 | 0,0162 |
| V26 | 0,0142 |
| V5 | 0,0135 |

Fitur `V14` dan `V10` paling dominan pada model lokal. Karena fitur PCA
dianonimkan, interpretasi bisnis langsung terbatas, tetapi fitur-fitur ini
menunjukkan sinyal numerik yang paling sering dimanfaatkan model.

## Perbandingan baseline

Baseline: Logistic Regression dengan `class_weight="balanced"`.

| Model | Threshold | Precision fraud | Recall fraud | F1 fraud | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression balanced | 0,5 | 0,0616 | 0,9184 | 0,1154 | 0,7209 |
| XGBoost default | 0,5 | 0,6774 | 0,8571 | 0,7568 | 0,8541 |
| XGBoost selected | 0,9 | 0,8333 | 0,8163 | 0,8247 | 0,8541 |

Logistic Regression memiliki recall lebih tinggi, tetapi false positive jauh
lebih besar. XGBoost memberi keseimbangan yang lebih baik antara recall dan
precision fraud.

## Kesimpulan

XGBoost memberikan performa yang kuat untuk dataset fraud detection yang sangat
imbalance. Threshold default 0,5 lebih cocok jika tujuan utama adalah menangkap
lebih banyak fraud. Threshold 0,9 lebih cocok jika false positive harus ditekan
tanpa menurunkan recall terlalu jauh.

Pengembangan lanjutan yang disarankan:

- Tuning hyperparameter XGBoost dengan Stratified K-Fold.
- Optimasi threshold berdasarkan biaya false negative dan false positive.
- Eksperimen SMOTE hanya pada training data sebagai pembanding.
- Monitoring data drift jika model dipakai dalam skenario production.
- Interpretasi SHAP untuk analisis kontribusi fitur yang lebih detail.

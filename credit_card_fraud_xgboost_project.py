# %% [markdown]
# # Credit Card Fraud Detection dengan XGBoost
#
# Project ini membangun model klasifikasi untuk mendeteksi transaksi fraud pada
# dataset Credit Card Fraud Detection. Target yang dipakai adalah `Class`:
#
# - `0`: transaksi normal
# - `1`: transaksi fraud
#
# Karena kelas fraud sangat sedikit, evaluasi difokuskan pada precision, recall,
# F1-score, ROC-AUC, dan terutama PR-AUC. Accuracy tidak dijadikan metrik utama.

# %% [markdown]
# ## 0. Setup library
#
# Jalankan cell instalasi di bawah ini hanya jika library belum tersedia,
# terutama saat memakai Google Colab.

# %%
# Jika dibutuhkan di Google Colab, hilangkan tanda komentar pada baris berikut:
# !pip install xgboost imbalanced-learn shap

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load dan pahami dataset
#
# Dataset diasumsikan berada di `data/creditcard.csv`. Jika dijalankan di Colab,
# unggah file `creditcard.csv` lalu sesuaikan `DATA_PATH` bila perlu.

# %%
DATA_PATH = Path("data/creditcard.csv")

if not DATA_PATH.exists():
    colab_path = Path("/content/creditcard.csv")
    if colab_path.exists():
        DATA_PATH = colab_path
    else:
        raise FileNotFoundError(
            "File dataset tidak ditemukan. Letakkan file CSV di "
            "`data/creditcard.csv` atau unggah `creditcard.csv` ke Colab."
        )

df = pd.read_csv(DATA_PATH)

print("Ukuran dataset:", df.shape)
print("\nNama kolom:")
print(df.columns.tolist())
print("\nTipe data:")
print(df.dtypes)

df.head()

# %% [markdown]
# **Interpretasi:** kolom `V1` sampai `V28` adalah fitur numerik hasil transformasi
# PCA. Kolom `Time` menunjukkan selisih waktu transaksi dari transaksi pertama
# pada dataset, sedangkan `Amount` menunjukkan nilai transaksi. Kolom `Class`
# adalah target klasifikasi.

# %%
class_counts = df["Class"].value_counts().sort_index()
class_percent = df["Class"].value_counts(normalize=True).sort_index() * 100
imbalance_summary = pd.DataFrame(
    {
        "class_label": ["Normal (0)", "Fraud (1)"],
        "count": class_counts.values,
        "percentage": class_percent.values,
    },
    index=class_counts.index,
)

display(imbalance_summary)

normal_count = class_counts.loc[0]
fraud_count = class_counts.loc[1]
imbalance_ratio = normal_count / fraud_count

print(f"Jumlah transaksi normal: {normal_count:,}")
print(f"Jumlah transaksi fraud : {fraud_count:,}")
print(f"Persentase normal      : {class_percent.loc[0]:.4f}%")
print(f"Persentase fraud       : {class_percent.loc[1]:.4f}%")
print(f"Rasio normal:fraud     : {imbalance_ratio:.2f}:1")

# %%
plt.figure(figsize=(6, 4))
ax = sns.barplot(
    x=class_counts.index.astype(str),
    y=class_counts.values,
    palette=["#4C78A8", "#F58518"],
)
ax.set_title("Distribusi Kelas Transaksi")
ax.set_xlabel("Class")
ax.set_ylabel("Jumlah transaksi")
ax.bar_label(ax.containers[0], fmt="%d")
plt.show()

# %% [markdown]
# **Interpretasi:** kelas fraud hanya sebagian sangat kecil dari dataset. Model
# yang hanya menebak "normal" hampir selalu benar secara accuracy, tetapi gagal
# untuk tujuan deteksi fraud.

# %% [markdown]
# ## 2. Exploratory Data Analysis
#
# Bagian ini mengecek missing value, duplikasi, statistik deskriptif, distribusi
# `Amount` dan `Time`, serta korelasi fitur terhadap target.

# %%
missing_values = df.isna().sum().sort_values(ascending=False)
duplicate_count = df.duplicated().sum()

print("Total missing value:", int(missing_values.sum()))
display(missing_values[missing_values > 0])
print("Jumlah baris duplikat:", int(duplicate_count))

# %%
df.describe().T

# %% [markdown]
# **Interpretasi:** tidak ada missing value pada dataset ini. Baris duplikat
# terdeteksi; untuk baseline yang umum pada dataset Credit Card Fraud Detection,
# baris tersebut tidak langsung dihapus agar hasil tetap sebanding dengan banyak
# referensi. Jika tujuan production, duplikasi perlu ditelusuri sebagai aturan
# bisnis terpisah.

# %%
amount_by_class = df.groupby("Class")["Amount"].describe()
time_by_class = df.groupby("Class")["Time"].describe()

print("Statistik Amount per kelas:")
display(amount_by_class)

print("Statistik Time per kelas:")
display(time_by_class)

# %%
plt.figure(figsize=(9, 4))
sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=80,
    stat="density",
    common_norm=False,
    element="step",
)
plt.xlim(0, df["Amount"].quantile(0.99))
plt.title("Distribusi Amount per Kelas (dibatasi sampai p99)")
plt.xlabel("Amount")
plt.ylabel("Density")
plt.show()

# %%
plot_df = df.assign(Time_hours=df["Time"] / 3600)

plt.figure(figsize=(9, 4))
sns.histplot(
    data=plot_df,
    x="Time_hours",
    hue="Class",
    bins=48,
    stat="density",
    common_norm=False,
    element="step",
)
plt.title("Distribusi Time per Kelas")
plt.xlabel("Time dari transaksi pertama (jam)")
plt.ylabel("Density")
plt.show()

# %% [markdown]
# **Interpretasi:** nilai `Amount` pada transaksi fraud tidak selalu besar;
# median fraud dapat lebih kecil dari transaksi normal. Ini berarti model perlu
# memakai pola multivariat dari fitur PCA, bukan hanya mengandalkan nominal
# transaksi. `Time` dapat memberi sinyal temporal, tetapi tetap harus dipakai
# dengan hati-hati karena dataset hanya mencakup rentang waktu tertentu.

# %%
corr = df.corr(numeric_only=True)
class_corr = corr["Class"].drop("Class").sort_values(key=lambda s: s.abs(), ascending=False)

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.1)
plt.title("Correlation Heatmap Semua Fitur")
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(
    x=class_corr.head(15).values,
    y=class_corr.head(15).index,
    palette="viridis",
)
plt.title("15 Korelasi Absolut Tertinggi terhadap Class")
plt.xlabel("Korelasi terhadap Class")
plt.ylabel("Fitur")
plt.show()

display(class_corr.head(15).to_frame("corr_with_class"))

# %% [markdown]
# **Interpretasi:** korelasi linear terhadap `Class` dapat membantu membaca sinyal
# awal, tetapi fraud detection sering membutuhkan model non-linear karena pola
# fraud muncul dari kombinasi fitur, bukan dari satu fitur tunggal.

# %% [markdown]
# ## 3. Preprocessing
#
# Fitur `V1` sampai `V28` tidak di-scaling ulang karena sudah merupakan hasil PCA.
# Scaling hanya diterapkan pada `Time` dan `Amount`. Scaler di-fit hanya pada data
# training untuk menghindari data leakage.
#
# Selain train-test split 80:20, data training dipecah lagi menjadi train dan
# validation. Validation dipakai untuk threshold tuning agar test set tetap
# menjadi evaluasi akhir yang lebih objektif.

# %%
X = df.drop(columns="Class")
y = df["Class"]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    stratify=y_train_full,
    random_state=RANDOM_STATE,
)

scaler = RobustScaler()
scale_cols = ["Time", "Amount"]

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

print("Ukuran train     :", X_train_scaled.shape)
print("Ukuran validation:", X_val_scaled.shape)
print("Ukuran test      :", X_test_scaled.shape)
print("\nDistribusi kelas train:")
print(y_train.value_counts().sort_index())
print("\nDistribusi kelas validation:")
print(y_val.value_counts().sort_index())
print("\nDistribusi kelas test:")
print(y_test.value_counts().sort_index())

# %% [markdown]
# **Interpretasi:** stratified split menjaga proporsi kelas fraud dan normal pada
# train, validation, dan test. Ini penting karena jumlah fraud sangat sedikit.

# %% [markdown]
# ## 4. Handling imbalanced data
#
# Untuk XGBoost, imbalance ditangani dengan `scale_pos_weight`, yaitu:
#
# `jumlah kelas normal pada training / jumlah kelas fraud pada training`

# %%
train_class_counts = y_train.value_counts().sort_index()
negative_count = train_class_counts.loc[0]
positive_count = train_class_counts.loc[1]
scale_pos_weight = negative_count / positive_count

print(f"Normal pada training: {negative_count:,}")
print(f"Fraud pada training : {positive_count:,}")
print(f"scale_pos_weight    : {scale_pos_weight:.4f}")

# %% [markdown]
# **Interpretasi:** bobot ini memberi penalti lebih besar pada kesalahan terhadap
# kelas fraud, sehingga model tidak terlalu condong ke kelas normal.

# %% [markdown]
# ## 5. Fungsi evaluasi
#
# Fungsi berikut dipakai untuk menghitung metrik utama. `F2-score` ditambahkan
# untuk threshold tuning karena F2 memberi bobot lebih besar pada recall.

# %%
def evaluate_at_threshold(y_true, y_proba, threshold=0.5):
    """Return metrics and confusion-matrix components for one threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "precision_fraud": precision_score(y_true, y_pred, zero_division=0),
        "recall_fraud": recall_score(y_true, y_pred, zero_division=0),
        "f1_fraud": f1_score(y_true, y_pred, zero_division=0),
        "f2_fraud": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_probability_model(name, y_true, y_proba, threshold=0.5):
    """Print classification metrics for a probability-based binary classifier."""
    metrics = evaluate_at_threshold(y_true, y_proba, threshold)
    y_pred = (y_proba >= threshold).astype(int)

    print(f"Model: {name}")
    print(f"Threshold: {threshold}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"ROC-AUC : {roc_auc_score(y_true, y_proba):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_true, y_proba):.4f}")

    return metrics


def plot_confusion_matrix(y_true, y_proba, threshold, title):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4.5, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Normal", "Pred Fraud"],
        yticklabels=["Actual Normal", "Actual Fraud"],
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

# %% [markdown]
# ## 6. Modeling dengan XGBoost
#
# Model utama menggunakan `XGBClassifier` dengan objective `binary:logistic`,
# `eval_metric="aucpr"`, dan `scale_pos_weight` dari data training.

# %%
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.90,
    colsample_bytree=0.90,
    reg_lambda=1.0,
    n_jobs=-1,
    tree_method="hist",
)

xgb_model.fit(X_train_scaled, y_train)

xgb_val_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
xgb_test_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

xgb_default_metrics = evaluate_probability_model(
    "XGBoost default threshold",
    y_test,
    xgb_test_proba,
    threshold=0.5,
)
plot_confusion_matrix(
    y_test,
    xgb_test_proba,
    threshold=0.5,
    title="XGBoost Confusion Matrix - Threshold 0.5",
)

# %% [markdown]
# **Interpretasi:** evaluasi threshold 0.5 menjadi baseline model XGBoost. Pada
# fraud detection, fokus utama adalah berapa banyak fraud yang berhasil ditemukan
# (`recall`) dan seberapa banyak alert fraud yang benar (`precision`).

# %% [markdown]
# ## 7. Threshold tuning
#
# Threshold diuji pada validation set, bukan test set, untuk mengurangi risiko
# data leakage dalam pemilihan threshold. Setelah threshold dipilih, performa
# akhir dievaluasi pada test set.

# %%
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_results = pd.DataFrame(
    [evaluate_at_threshold(y_val, xgb_val_proba, threshold) for threshold in thresholds]
)

display(threshold_results)

# Pilih threshold dengan F2-score tertinggi karena F2 lebih memprioritaskan recall.
best_threshold_row = threshold_results.sort_values(
    ["f2_fraud", "recall_fraud", "precision_fraud"],
    ascending=False,
).iloc[0]
best_threshold = float(best_threshold_row["threshold"])

print("Threshold terbaik berdasarkan validation F2-score:")
display(best_threshold_row.to_frame("value"))

# %%
xgb_best_metrics = evaluate_probability_model(
    "XGBoost selected threshold",
    y_test,
    xgb_test_proba,
    threshold=best_threshold,
)
plot_confusion_matrix(
    y_test,
    xgb_test_proba,
    threshold=best_threshold,
    title=f"XGBoost Confusion Matrix - Threshold {best_threshold}",
)

# %% [markdown]
# **Interpretasi trade-off:** threshold rendah biasanya meningkatkan recall
# fraud, tetapi menambah false positive. Threshold tinggi biasanya menurunkan
# false positive dan meningkatkan precision, tetapi bisa melewatkan lebih banyak
# fraud. Pada kasus fraud detection, threshold operasional sebaiknya dipilih
# berdasarkan kapasitas tim review dan toleransi risiko fraud yang lolos.

# %% [markdown]
# ## 8. Feature importance XGBoost
#
# Feature importance berikut berasal dari model XGBoost. Nilai tinggi berarti
# fitur tersebut sering dipakai model untuk memisahkan kelas, tetapi bukan bukti
# hubungan kausal.

# %%
feature_importance = pd.DataFrame(
    {
        "feature": X_train_scaled.columns,
        "importance": xgb_model.feature_importances_,
    }
).sort_values("importance", ascending=False)

display(feature_importance.head(15))

plt.figure(figsize=(8, 6))
sns.barplot(
    data=feature_importance.head(15),
    x="importance",
    y="feature",
    palette="mako",
)
plt.title("Top 15 Feature Importance - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# %% [markdown]
# **Interpretasi:** fitur teratas menunjukkan sinyal paling kuat menurut model.
# Karena `V1` sampai `V28` adalah hasil PCA dan sudah dianonimkan, penjelasan
# bisnis langsung terbatas. Namun, ranking ini tetap berguna untuk memahami pola
# teknis yang dipakai model.

# %% [markdown]
# ### Interpretasi dengan SHAP (opsional)
#
# SHAP memberi interpretasi kontribusi fitur terhadap prediksi. Cell ini memakai
# sampel test agar waktu komputasi tetap ringan. Jika library SHAP tidak tersedia
# atau tidak kompatibel dengan versi XGBoost lokal, analisis feature importance di
# atas tetap dapat dipakai.

# %%
try:
    import shap

    shap_sample = X_test_scaled.sample(
        n=min(1000, len(X_test_scaled)),
        random_state=RANDOM_STATE,
    )
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(shap_sample)

    shap.summary_plot(
        shap_values,
        shap_sample,
        max_display=15,
        show=True,
    )
except Exception as exc:
    print("SHAP tidak dapat dijalankan di environment ini.")
    print("Alasan:", repr(exc))

# %% [markdown]
# ## 9. Baseline: Logistic Regression
#
# Sebagai pembanding sederhana, digunakan Logistic Regression dengan
# `class_weight="balanced"`. Model ini membantu melihat apakah XGBoost memberi
# nilai tambah pada data tabular imbalance.

# %%
baseline_lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
    random_state=RANDOM_STATE,
)

baseline_lr.fit(X_train_scaled, y_train)
lr_test_proba = baseline_lr.predict_proba(X_test_scaled)[:, 1]

lr_metrics = evaluate_probability_model(
    "Logistic Regression balanced",
    y_test,
    lr_test_proba,
    threshold=0.5,
)
plot_confusion_matrix(
    y_test,
    lr_test_proba,
    threshold=0.5,
    title="Logistic Regression Confusion Matrix - Threshold 0.5",
)

# %%
comparison = pd.DataFrame(
    [
        {
            "model": "Logistic Regression balanced",
            "threshold": 0.5,
            "precision_fraud": lr_metrics["precision_fraud"],
            "recall_fraud": lr_metrics["recall_fraud"],
            "f1_fraud": lr_metrics["f1_fraud"],
            "pr_auc": average_precision_score(y_test, lr_test_proba),
        },
        {
            "model": "XGBoost default threshold",
            "threshold": 0.5,
            "precision_fraud": xgb_default_metrics["precision_fraud"],
            "recall_fraud": xgb_default_metrics["recall_fraud"],
            "f1_fraud": xgb_default_metrics["f1_fraud"],
            "pr_auc": average_precision_score(y_test, xgb_test_proba),
        },
        {
            "model": "XGBoost selected threshold",
            "threshold": best_threshold,
            "precision_fraud": xgb_best_metrics["precision_fraud"],
            "recall_fraud": xgb_best_metrics["recall_fraud"],
            "f1_fraud": xgb_best_metrics["f1_fraud"],
            "pr_auc": average_precision_score(y_test, xgb_test_proba),
        },
    ]
)

display(comparison.sort_values("pr_auc", ascending=False))

# %% [markdown]
# **Interpretasi:** Logistic Regression sering menghasilkan recall tinggi, tetapi
# dengan false positive jauh lebih banyak. XGBoost biasanya lebih baik untuk data
# tabular non-linear karena mampu menangkap interaksi antar fitur PCA.

# %% [markdown]
# ## 10. Ringkasan akhir
#
# Cell berikut mencetak ringkasan utama agar hasil project mudah dibaca ulang
# setelah notebook dijalankan.

# %%
print("Ringkasan dataset")
print(f"- Total data: {len(df):,} transaksi")
print(f"- Normal: {normal_count:,} ({class_percent.loc[0]:.4f}%)")
print(f"- Fraud : {fraud_count:,} ({class_percent.loc[1]:.4f}%)")
print(f"- Missing value total: {int(df.isna().sum().sum())}")
print(f"- Duplikasi: {int(df.duplicated().sum())}")

print("\nAlasan XGBoost")
print("- Cocok untuk data tabular numerik.")
print("- Mampu menangkap hubungan non-linear antar fitur.")
print("- Mendukung handling imbalance melalui scale_pos_weight.")

print("\nPreprocessing")
print("- X dan y dipisah dari kolom Class.")
print("- Time dan Amount di-scale dengan RobustScaler.")
print("- V1 sampai V28 tidak di-scale ulang karena merupakan fitur PCA.")
print("- Split dilakukan secara stratified.")
print("- Threshold dipilih menggunakan validation set.")

print("\nHasil XGBoost pada test set")
print(f"- Threshold default 0.5 recall fraud: {xgb_default_metrics['recall_fraud']:.4f}")
print(f"- Threshold default 0.5 precision fraud: {xgb_default_metrics['precision_fraud']:.4f}")
print(f"- Threshold terpilih: {best_threshold}")
print(f"- Recall fraud threshold terpilih: {xgb_best_metrics['recall_fraud']:.4f}")
print(f"- Precision fraud threshold terpilih: {xgb_best_metrics['precision_fraud']:.4f}")
print(f"- F1 fraud threshold terpilih: {xgb_best_metrics['f1_fraud']:.4f}")
print(f"- ROC-AUC: {roc_auc_score(y_test, xgb_test_proba):.4f}")
print(f"- PR-AUC : {average_precision_score(y_test, xgb_test_proba):.4f}")

print("\nSaran pengembangan")
print("- Validasi dengan cross-validation berbasis stratified fold.")
print("- Tuning hyperparameter XGBoost dengan fokus PR-AUC dan recall fraud.")
print("- Coba threshold sesuai kapasitas tim investigasi fraud.")
print("- Evaluasi cost-sensitive learning bila biaya false negative dan false positive diketahui.")
print("- Uji monitoring data drift untuk skenario production.")

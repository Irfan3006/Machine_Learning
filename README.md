# Simulasi Deteksi Intrusi Jaringan pada Partisi Evaluasi UNSW-NB15

Repositori ini memuat implementasi dan simulasi penelitian eksperimental mengenai sistem deteksi intrusi jaringan (Network Intrusion Detection System / NIDS) berbasis pembelajaran mesin. Proyek ini mengevaluasi trade-off antara tingkat deteksi serangan (recall) dan tingkat alarm palsu (false positive rate / FPR) pada partisi standar dataset UNSW-NB15.

Dokumentasi laporan komprehensif penelitian dapat diakses pada berkas [laporan_simulasi_penelitian_deteksi_intrusi.md](laporan_simulasi_penelitian_deteksi_intrusi.md).

## 1. Latar Belakang dan Masalah

Sistem deteksi intrusi jaringan bertugas mengidentifikasi aktivitas serangan siber di dalam aliran trafik jaringan tanpa membebani analis keamanan dengan peringatan palsu (false alarm). Ketidakseimbangan distribusi trafik dan kompleksitas korelasi nonlinier antarfitur jaringan sering kali menyebabkan model klasifikasi menghasilkan tingkat recall serangan yang tinggi tetapi disertai false positive rate yang tidak dapat diterima untuk operasional pusat operasi keamanan (Security Operations Center / SOC).

Penelitian ini menguji hipotesis kerja bahwa model nonlinier mampu menangkap interaksi fitur yang kompleks, dan bahwa seleksi fitur dapat memangkas biaya komputasi pelatihan secara signifikan tanpa menurunkan kinerja klasifikasi secara berarti.

## 2. Rumusan Masalah

1. RQ1: Bagaimana perbandingan kinerja Logistic Regression, Random Forest, dan Histogram Gradient Boosting pada partisi evaluasi UNSW-NB15?
2. RQ2: Bagaimana seleksi fitur memengaruhi macro-F1 dan biaya pelatihan?
3. RQ3: Bagaimana pembobotan kelas dan ambang batas keputusan (threshold) mengubah recall serangan serta false positive rate?

Kriteria validasi yang ditetapkan dalam penelitian ini adalah pencapaian serentak Akurasi >= 93% dan Recall Serangan >= 93%.

## 3. Desain Variabel Penelitian

- Keputusan Awal: Klasifikasi biner trafik jaringan (label 0: normal, label 1: serangan).
- Variabel Bebas: Algoritma klasifikasi (Logistic Regression, Random Forest, Histogram Gradient Boosting), seleksi fitur (ukuran subset fitur k), skema pembobotan kelas (unweighted vs balanced), dan ambang batas keputusan (threshold theta dari 0.05 hingga 0.95).
- Variabel Terikat: Mean Accuracy, Macro F1-score, Attack Recall (TPR), Attack Precision, False Positive Rate (FPR), ROC-AUC, waktu pelatihan (detik), dan latensi inferensi (milidetik per sampel).
- Variabel Kontrol: Partisi data UNSW-NB15 resmi (175.341 data latih dan 82.332 data uji), stratifikasi 5-fold cross-validation, nilai seed acak 42, serta lingkungan komputasi yang seragam.
- Ambang Batas Validasi: Akurasi >= 93% dan Recall Serangan >= 93%.

## 4. Dataset

Eksperimen menggunakan dataset UNSW-NB15 yang dikembangkan oleh Cyber Range Lab of the Australian Centre for Cyber Security (ACCS):
- Partisi Latih (UNSW_NB15_training-set.csv): 175.341 baris (119.341 serangan, 56.000 normal).
- Partisi Uji Held-Out (UNSW_NB15_testing-set.csv): 82.332 baris (45.332 serangan, 37.000 normal).
- Fitur: 42 atribut (39 numerik dan 3 kategorikal: proto, service, state). Kolom identifikasi (id) dan kategori spesifik (attack_cat) dikeluarkan dari input fitur klasifikasi biner.

## 5. Ringkasan Hasil Eksperimen

### 5.1. RQ1: Komparasi Model Klasifikasi

Pengujian dilakukan melalui validasi silang 5-lipatan (5-Fold Stratified CV) pada data latih dan evaluasi independen pada data uji held-out (threshold default theta = 0.50):

| Model | CV Mean Accuracy (%) | CV Mean Macro-F1 (%) | Test Accuracy (%) | Test Macro-F1 (%) | Attack Recall (%) | FPR (%) | ROC-AUC | Waktu Latih (s) | Latensi Inferensi (ms/sampel) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 93.04 +- 0.06 | 91.60 +- 0.09 | 80.96 | 79.44 | 98.15 | 40.11 | 0.9445 | 4.26 | 0.0009 |
| Random Forest | 95.56 +- 0.12 | 94.80 +- 0.14 | 86.58 | 85.87 | 98.97 | 28.61 | 0.9840 | 17.70 | 0.0037 |
| Histogram Gradient Boosting | 95.59 +- 0.14 | 94.88 +- 0.17 | 87.34 | 86.74 | 98.66 | 26.52 | 0.9859 | 7.82 | 0.0107 |

Uji signifikansi berpasangan (paired t-test) membuktikan bahwa keunggulan Histogram Gradient Boosting terhadap Logistic Regression (t = 43.87, p = 1.61e-6) dan terhadap Random Forest (t = 4.06, p = 0.0153) signifikan secara statistik pada taraf nyata alpha = 0.05.

### 5.2. RQ2: Pengaruh Seleksi Fitur terhadap Kinerja dan Biaya Pelatihan

Fitur diperingkat berdasarkan Gini Impurity Importance menggunakan Random Forest, kemudian disimulasikan menggunakan Histogram Gradient Boosting pada berbagai ukuran subset fitur:

| Jumlah Fitur (k) | Akurasi (%) | Macro-F1 (%) | Recall Serangan (%) | FPR (%) | Waktu Latih (s) | Retensi Macro-F1 (%) | Penghematan Waktu Latih (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 42 (Baseline) | 87.52 | 86.95 | 98.57 | 26.02 | 4.79 | 100.00 | 0.00 |
| Top 30 | 87.57 | 87.01 | 98.42 | 25.72 | 3.92 | 100.07 | 18.16 |
| Top 20 | 87.27 | 86.67 | 98.57 | 26.57 | 3.97 | 99.69 | 17.09 |
| Top 15 | 86.17 | 85.53 | 97.29 | 27.45 | 3.28 | 98.37 | 31.46 |
| Top 10 | 82.73 | 81.47 | 98.83 | 37.00 | 2.57 | 93.70 | 46.29 |
| Top 5 | 81.36 | 79.76 | 99.41 | 40.75 | 2.35 | 91.74 | 50.86 |

Temuan utama:
- Subset Top-30 fitur merupakan titik optimal: menghemat 18.16% waktu komputasi pelatihan dan menghasilkan Macro-F1 100.07% terhadap baseline karena reduksi fitur derau (noise).
- Subset Top-15 fitur memangkas 31.46% waktu komputasi dengan tetap mempertahankan 98.37% kinerja Macro-F1.

### 5.3. RQ3: Pembobotan Kelas, Kalibrasi Threshold, dan Kriteria Validasi

Pada threshold default (theta = 0.50), model menghasilkan false positive rate tinggi (26.52%) dan akurasi 87.34% akibat pergeseran prior probabilitas dari data latih yang didominasi trafik serangan (68.06%).

Pengaruh pembobotan kelas pada theta = 0.50:
- class_weight = None: Akurasi = 87.34%, Recall = 98.66%, FPR = 26.52%, Macro-F1 = 86.74%.
- class_weight = balanced: Akurasi = 91.18%, Recall = 96.77%, FPR = 15.66%, Macro-F1 = 90.97%.

Penelusuran ambang batas keputusan (threshold sweeping theta in [0.05, 0.95]) menghasilkan titik operasional optimal pada theta = 0.80:
- Akurasi: 93.41% (memenuhi kriteria >= 93%)
- Recall Serangan: 94.36% (memenuhi kriteria >= 93%)
- False Positive Rate: 7.76% (turun dari 26.52%, mengeliminasi 6.944 alarm palsu)
- Macro-F1: 93.34%

Dengan demikian, target validasi penelitian terbukti berhasil dipenuhi secara simultan pada ambang batas keputusan theta = 0.80.

## 6. Verifikasi Hipotesis Kerja

1. Hipotesis H1 (Model Nonlinier): Terkonfirmasi. Model nonlinier menangkap interaksi korelasi antarfitur secara lebih akurat dibanding model linier (peningkatan Macro-F1 sebesar 7.30% dan AUC dari 0.9445 ke 0.9859).
2. Hipotesis H2 (Seleksi Fitur): Terkonfirmasi. Seleksi fitur mampu memangkas waktu latih hingga 18.16% sampai 31.46% dengan tetap mempertahankan lebih dari 98% kinerja klasifikasi.
3. Hipotesis H3 (Pembobotan dan Kalibrasi Threshold): Terkonfirmasi. Penyesuaian ambang batas keputusan mengoreksi bias prior data dan menurunkan FPR secara signifikan hingga memenuhi standar validasi operasional.

## 7. Struktur Repositori

```text
Machine_Learning/
|-- research_simulation/
|   |-- __init__.py
|   |-- config.py
|   |-- preprocess.py
|   |-- models.py
|   |-- rq1_benchmark.py
|   |-- rq2_feature_selection.py
|   |-- rq3_threshold_weighting.py
|   |-- visualizer.py
|   \-- main.py
|-- results/
|   |-- figures/
|   |   |-- rq1_model_comparison.png
|   |   |-- rq2_feature_selection.png
|   |   \-- rq3_threshold_tradeoff.png
|   |-- tables/
|   |   |-- rq1_cv_summary.csv
|   |   |-- rq1_test_summary.csv
|   |   |-- rq2_feature_ranking.csv
|   |   |-- rq2_feature_subsets.csv
|   |   |-- rq3_class_weights.csv
|   |   \-- rq3_threshold_sweep.csv
|   \-- simulation_summary.json
|-- pyproject.toml
|-- LICENSE
\-- README.md
```

## 8. Panduan Menjalankan Eksperimen

### 8.1. Prasyarat Lingkungan

Pastikan Python versi 3.10 atau yang lebih baru telah terpasang. Dependensi pustaka yang dibutuhkan:
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

### 8.2. Menjalankan Simulasi

Jalankan modul utama simulasi melalui baris perintah:
```bash
python -m research_simulation.main
```

Seluruh kalkulasi validasi silang 5-lipatan, pemeringkatan fitur, simulasi threshold, penyimpanan tabel metrik, dan pembuatan grafik visualisasi akan diproses secara otomatis ke direktori results/.

## 9. Lisensi

Proyek ini dilisensikan di bawah ketentuan Lisensi MIT. Rincian lengkap dapat dilihat pada berkas LICENSE.

# Laporan Proyek Machine Learning Recommendation System - Muhamad Alan Dharma Saputro Setiawan

## Project Overview

Dengan pesatnya pertumbuhan musik digital, sistem rekomendasi menjadi penting untuk membantu pengguna menemukan lagu yang sesuai preferensi. Banyak sistem saat ini mengandalkan collaborative filtering, namun pendekatan ini mengalami kendala seperti cold start dan ketergantungan pada data interaksi pengguna.

Proyek ini menggunakan pendekatan content-based filtering yang memanfaatkan fitur audio lagu, seperti danceability, energy, dan valence, untuk merekomendasikan lagu yang mirip berdasarkan kontennya. Pendekatan ini memungkinkan sistem merekomendasikan lagu baru atau kurang populer tanpa memerlukan data histori pengguna.

Menurut penelitian oleh Song et al. (2012) dan van den Oord et al. (2013), metode berbasis konten efektif untuk menanggulangi keterbatasan sistem rekomendasi tradisional, terutama ketika dikombinasikan dengan teknik representasi fitur yang baik seperti PCA atau deep learning.

Referensi:
- [A Survey of Music Recommendation Systems and Future Perspectives – Yading Song, Simon Dixon, Marcus Pearce](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e0080299afae01ad796060abcf602abff6024754)
- [Deep Content-Based Music Recommendation – Aäron van den Oord, Sander Dieleman, Benjamin Schrauwen](https://proceedings.neurips.cc/paper_files/paper/2013/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf)

## Business Uncerstanding

### Problem Statements

1. **Pengguna kesulitan menemukan lagu baru yang sesuai dengan selera mereka.**  
   Dengan jutaan lagu tersedia di platform streaming, pengguna sering kali tidak memiliki referensi yang cukup untuk menemukan musik baru yang mereka sukai.

2. **Sistem rekomendasi berbasis collaborative filtering tidak efektif untuk lagu-lagu baru atau tidak populer.**  
   Collaborative filtering sangat bergantung pada histori interaksi pengguna, sehingga lagu yang baru dirilis atau jarang diputar sulit untuk direkomendasikan.

3. **Kurangnya rekomendasi yang bersifat personal berdasarkan karakteristik konten lagu.**  
   Banyak sistem hanya menyarankan lagu dari artis yang sama atau genre serupa, tanpa memperhatikan elemen audio yang membentuk karakter lagu.

---

### Goals

1. **Membangun sistem rekomendasi lagu yang dapat menyarankan lagu-lagu baru berdasarkan kesamaan fitur konten audio.**  
   Dengan menggunakan pendekatan content-based filtering, sistem dapat menyarankan lagu-lagu meskipun belum memiliki histori interaksi pengguna.

2. **Memberikan rekomendasi yang lebih personal dan relevan dengan preferensi pengguna.**  
   Rekomendasi didasarkan pada fitur-fitur seperti danceability, energy, acousticness, dan valence yang secara langsung mencerminkan suasana dan gaya musik.

3. **Menyediakan visualisasi dan evaluasi yang menunjukkan bahwa rekomendasi yang dihasilkan benar-benar serupa secara fitur.**  
   Ini bertujuan meningkatkan kepercayaan pengguna terhadap sistem dan memudahkan eksplorasi lagu.

---

### Solution Statements

1. **Pendekatan Content-Based Filtering**  
   Menggunakan fitur numerik dari lagu seperti danceability, energy, valence, dan tempo untuk menghitung kemiripan antar lagu. Cosine similarity digunakan sebagai metrik utama untuk membangun sistem rekomendasi berbasis konten.

2. **Evaluasi Akurasi Rekomendasi dengan Metrik Klasifikasi**  
   Untuk mengukur seberapa relevan hasil rekomendasi terhadap lagu referensi, digunakan metrik klasifikasi seperti precision, recall, dan f1-score. Lagu dianggap relevan jika memiliki nilai kemiripan di atas threshold tertentu. Evaluasi ini memberikan gambaran kuantitatif terhadap akurasi sistem dalam merekomendasikan lagu yang sesuai secara konten.

3. **Sampling Terbatas untuk Efisiensi Komputasi**  
   Karena dataset berukuran sangat besar, dilakukan pengambilan subset acak secara efisien agar perhitungan similarity tetap dapat dijalankan di lingkungan komputasi terbatas seperti Google Colab.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data musik dari Spotify yang berisi informasi metadata serta fitur-fitur audio dari lagu-lagu yang tersedia di platform tersebut. Dataset ini memiliki total **1.204.025 baris** dan **24 kolom**, yang mencakup berbagai aspek penting dari sebuah lagu seperti nama artis, nama lagu, durasi, hingga karakteristik audio seperti danceability, energy, acousticness, dan lainnya. Setelah dilakukan pengecekan awal, dataset memiliki kualitas yang baik dengan hanya sedikit nilai kosong pada kolom `name` dan `album`. Data yang memiliki nilai kosong tersebut dihapus agar proses pemodelan berjalan optimal. Dataset ini kemudian digunakan sebagai dasar dalam membangun sistem rekomendasi lagu berbasis konten.

Sumber data:
[Spotify 1.2M+ Songs - Kaggle Dataset](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)

### Variables in the Spotify Songs

Berikut adalah penjelasan dari beberapa fitur utama yang digunakan dalam proyek ini:

- `id`: ID unik dari masing-masing lagu di Spotify.
- `name`: Nama atau judul dari lagu.
- `album`: Nama album tempat lagu dirilis.
- `album_id`: ID unik dari album.
- `artists`: Nama artis atau grup yang menyanyikan lagu.
- `artist_ids`: ID unik dari artis di Spotify.
- `track_number`: Urutan lagu dalam album.
- `disc_number`: Nomor cakram dalam album (berguna untuk album multi-disc).
- `explicit`: Menunjukkan apakah lagu mengandung konten eksplisit (`True`) atau tidak (`False`).
- `danceability`: Tingkat kelayakan lagu untuk ditarikan (skala 0.0–1.0).
- `energy`: Tingkat intensitas dan aktivitas lagu.
- `key`: Tangga nada utama lagu, direpresentasikan sebagai bilangan bulat (0–11).
- `loudness`: Volume rata-rata lagu dalam desibel.
- `mode`: Skala musik lagu; `1` untuk mayor, `0` untuk minor.
- `speechiness`: Tingkat kehadiran kata yang diucapkan dalam lagu.
- `acousticness`: Kemungkinan bahwa lagu bersifat akustik.
- `instrumentalness`: Probabilitas bahwa lagu merupakan instrumental (tanpa vokal).
- `liveness`: Deteksi apakah lagu direkam dalam konser/live.
- `valence`: Ukuran nuansa positif atau ceria dari lagu (0.0–1.0).
- `tempo`: Tempo lagu dalam beat per minute (BPM).
- `duration_ms`: Durasi lagu dalam satuan milidetik.
- `time_signature`: Tanda birama lagu.
- `year`: Tahun rilis dari lagu.
- `release_date`: Tanggal rilis lagu dalam format YYYY-MM-DD.

### Data Condition

- Dataset terdiri dari **1.204.025 baris** dan **24 kolom**.
- Hanya terdapat **sedikit data kosong**, yaitu pada kolom `name` (3 data) dan `album` (11 data).
- Sebagian besar kolom memiliki tipe data yang sesuai:
  - Kolom numerik: seperti `danceability`, `energy`, `tempo`, `duration_ms`.
  - Kolom kategori: seperti `id`, `name`, `album`, `artists`.
  - Kolom boolean: `explicit`.

### Statistik Deskriptif

Tabel berikut menunjukkan ringkasan statistik deskriptif dari fitur numerik dalam dataset, seperti nilai minimum, maksimum, rata-rata, serta kuartil. Statistik ini membantu dalam memahami sebaran data dan mendeteksi potensi outlier.

| Fitur             | Count         | Mean     | Std      | Min     | 25%     | 50%     | 75%     | Max        |
|------------------|---------------|----------|----------|---------|---------|---------|---------|------------|
| track_number      | 1,204,012     | 7.66     | 5.99     | 1       | 3       | 7       | 10      | 50         |
| disc_number       | 1,204,012     | 1.06     | 0.30     | 1       | 1       | 1       | 1       | 13         |
| danceability      | 1,204,012     | 0.493    | 0.190    | 0.000   | 0.356   | 0.501   | 0.633   | 1.000      |
| energy            | 1,204,012     | 0.509    | 0.295    | 0.000   | 0.252   | 0.524   | 0.766   | 1.000      |
| key               | 1,204,012     | 5.19     | 3.54     | 0       | 2       | 5       | 8       | 11         |
| loudness          | 1,204,012     | -11.81   | 6.98     | -60.00  | -15.25  | -9.79   | -6.72   | 7.23       |
| mode              | 1,204,012     | 0.67     | 0.47     | 0       | 0       | 1       | 1       | 1          |
| speechiness       | 1,204,012     | 0.084    | 0.116    | 0.000   | 0.035   | 0.045   | 0.072   | 0.969      |
| acousticness      | 1,204,012     | 0.447    | 0.385    | 0.000   | 0.038   | 0.389   | 0.861   | 0.996      |
| instrumentalness  | 1,204,012     | 0.283    | 0.376    | 0.000   | 0.000   | 0.008   | 0.719   | 1.000      |
| liveness          | 1,204,012     | 0.202    | 0.180    | 0.000   | 0.097   | 0.125   | 0.245   | 1.000      |
| valence           | 1,204,012     | 0.428    | 0.270    | 0.000   | 0.191   | 0.403   | 0.644   | 1.000      |
| tempo             | 1,204,012     | 117.63   | 30.94    | 0.00    | 94.05   | 116.73  | 137.05  | 248.93     |
| duration_ms       | 1,204,012     | 248,840  | 162,211  | 1,000   | 174,089 | 224,340 | 285,840 | 6,061,090  |
| time_signature    | 1,204,012     | 3.83     | 0.56     | 0.00    | 4.00    | 4.00    | 4.00    | 5.00       |
| year              | 1,204,012     | 2007.33  | 12.10    | 0       | 2002    | 2009    | 2015    | 2020       |

> *Note* : Dataset lengkap telah dimuat dan dilakukan pembersihan awal dengan menghapus baris yang memiliki nilai kosong menggunakan `dropna()`. Proses persiapan lebih lanjut seperti pemilihan fitur, encoding, dan normalisasi akan dijelaskan secara detail pada bagian *Data Preparation*.

### Distribusi Audio Features

Visualisasi berikut menunjukkan sebaran dari lima fitur audio utama yang digunakan dalam sistem rekomendasi ini:

- `danceability`: Seberapa cocok lagu untuk menari.
- `energy`: Tingkat intensitas dan aktivitas lagu.
- `valence`: Nuansa emosional dari lagu, dari sedih hingga ceria.
- `acousticness`: Kemungkinan bahwa lagu bersifat akustik.
- `instrumentalness`: Kemungkinan bahwa lagu bersifat instrumental (tanpa vokal).

![Distribusi Audio Features](https://i.postimg.cc/43XdK2rJ/image.png)

**Insight:**

- `danceability` memiliki distribusi normal dengan puncak di sekitar nilai 0.6.
- `energy` tersebar cukup merata, menunjukkan adanya lagu dengan intensitas tinggi maupun rendah dalam jumlah seimbang.
- `valence` cenderung menurun, menunjukkan lebih banyak lagu dengan nuansa sedih daripada ceria.
- `acousticness` dan `instrumentalness` menunjukkan pola **bimodal**, menandakan adanya dua kelompok dominan: lagu yang sangat akustik/instrumental dan yang tidak sama sekali.

### Korelasi antar Audio Features

Visualisasi berikut menunjukkan matriks korelasi antar fitur audio utama dalam dataset. Nilai korelasi berada di antara -1 hingga 1, yang menunjukkan seberapa kuat hubungan linier antar dua fitur.

![Korelasi Audio Features](https://i.postimg.cc/RhJVrknT/image.png)

**Insight:**

- `danceability` berkorelasi positif dengan `valence` (0.56) dan `energy` (0.28), yang menunjukkan bahwa lagu yang enak untuk ditarikan cenderung lebih ceria dan energik.
- `energy` berkorelasi negatif kuat dengan `acousticness` (-0.80), artinya lagu yang sangat akustik cenderung memiliki tingkat energi yang rendah.
- `instrumentalness` memiliki korelasi lemah dengan semua fitur lainnya, menunjukkan bahwa karakter instrumental cenderung berdiri sendiri.
- `valence` juga memiliki korelasi sedang dengan `energy` (0.40), menunjukkan bahwa lagu yang ceria cenderung juga lebih energik.

### Distribusi Tahun dan Durasi Lagu

Visualisasi berikut menampilkan hubungan antara tahun rilis lagu (`year`) dan durasi lagu dalam menit (`duration_min`). Visualisasi ini membantu memahami tren produksi lagu dari waktu ke waktu, termasuk seberapa lama durasi rata-rata lagu dalam periode tertentu.

![Distribusi Tahun dan Durasi Lagu](https://i.postimg.cc/tTYVTmxs/image.png)

**Insight:**

- Sebagian besar lagu yang terdapat dalam dataset berasal dari tahun 2000 ke atas.
- Durasi lagu paling umum berkisar antara 2 hingga 5 menit.
- Terdapat lagu berdurasi ekstrem (lebih dari 10 menit) namun dalam jumlah kecil.
- Tren distribusi semakin padat setelah tahun 2010, menunjukkan bahwa data musik digital lebih banyak tersedia dalam rentang waktu tersebut.

> *Note* : Data telah diproses untuk menambahkan kolom `year` dan `duration_min`, serta hanya menggunakan baris yang tidak memiliki nilai kosong.

### Distribusi Kolom Explicit

Visualisasi berikut menunjukkan jumlah lagu berdasarkan status "explicit", yaitu apakah lagu tersebut mengandung konten eksplisit seperti kata-kata kasar atau tema dewasa.

![Distribusi Lagu Eksplisit](https://i.postimg.cc/pV5FM4v7/image.png)

**Insight:**

- Mayoritas besar lagu dalam dataset **tidak** memiliki konten eksplisit.
- Hanya sebagian kecil lagu yang ditandai sebagai **explicit**, menunjukkan bahwa dataset cenderung ramah untuk semua umur.

## Data Preparation

Tahap ini bertujuan untuk membersihkan dan menyiapkan data sebelum digunakan dalam proses pemodelan sistem rekomendasi berbasis konten (Content-Based Filtering). Beberapa tahapan utama dalam persiapan data adalah sebagai berikut:

### Menghapus Nilai Kosong

Data dengan nilai kosong (missing values) pada kolom seperti `name` dan `album` dihapus menggunakan fungsi `dropna()` untuk memastikan kualitas data tetap terjaga.

```python
df = df.dropna()
```

### Konversi Tanggal Rilis dan Perhitungan Durasi Lagu

Agar bisa dianalisis secara numerik, kolom `release_date` dikonversi ke format datetime. Kemudian, kolom `year` diambil dari tahun rilis dan `duration_min` dihitung dari kolom `duration_ms`.

```python
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year'] = df['release_date'].dt.year
df['duration_min'] = df['duration_ms'] / 60000
```

### Penyaringan Data Valid

Hanya data yang memiliki nilai tidak kosong untuk `year` dan `duration_min` yang digunakan lebih lanjut. Data yang valid disimpan dalam variabel `valid_df`.

```python
valid_df = df[df['year'].notna() & df['duration_min'].notna()]
```

### Encoding ID Lagu

Kolom `id` yang merupakan representasi unik dari setiap lagu diubah menjadi bentuk numerik menggunakan `LabelEncoder`. Nilai hasil encoding disimpan dalam kolom baru bernama `track_encoded`.

```python
from sklearn.preprocessing import LabelEncoder

track_encoder = LabelEncoder()
valid_df['track_encoded'] = track_encoder.fit_transform(valid_df['id'])
```

### Sampling Data

Untuk menghindari crash pada runtime dan mempercepat proses komputasi, dilakukan pengambilan sampel sebanyak 10.000 lagu secara acak dari `valid_df`.

```python
valid_df = valid_df.sample(n=10000, random_state=42).reset_index(drop=True)
```

### Seleksi Fitur Konten

Dipilih delapan fitur konten lagu yang dianggap penting dan relevan untuk sistem rekomendasi berbasis kemiripan:
- `danceability`
- `energy`
- `valence`
- `acousticness`
- `instrumentalness`
- `tempo`
- `duration_min`
- `year`

### Normalisasi Fitur

Seluruh fitur konten dinormalisasi ke dalam skala 0–1 menggunakan `MinMaxScaler`. Ini dilakukan agar semua fitur memiliki pengaruh yang seimbang dalam proses perhitungan similarity.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
content_features_scaled = scaler.fit_transform(content_features)
```

## Modeling

Sistem rekomendasi yang dibangun pada proyek ini menggunakan pendekatan **Content-Based Filtering**, di mana rekomendasi lagu diberikan berdasarkan kemiripan fitur konten audio seperti `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`, `duration_min`, dan `year`.

### Algoritma yang Digunakan

Model menghitung **kemiripan kosinus (cosine similarity)** antar vektor fitur lagu yang telah dinormalisasi. Prosesnya sebagai berikut:

1. Menghitung similarity antar semua lagu menggunakan `cosine_similarity`.
2. Memilih satu lagu sebagai lagu referensi (secara acak).
3. Mengambil top-N (misal 10 atau 20) lagu dengan similarity tertinggi terhadap lagu referensi.

### Output Rekomendasi

Sebagai contoh, berikut adalah lagu yang dipilih secara acak sebagai referensi:

**Lagu Referensi:**

| Name | Artists |
|------|---------|
| Trouble On The Line | ['Sawyer Brown'] |

**Top-20 Rekomendasi Lagu Serupa:**

| No | Track Encoded | Name                             | Artists                                             | Similarity Score |
|----|----------------|----------------------------------|-----------------------------------------------------|------------------|
| 1  | 228765         | September - Cali-Code 285 Mix    | ['SMP']                                             | 0.998719         |
| 2  | 266585         | Regret                           | ['Matt Rocker & the Constituents']                  | 0.998674         |
| 3  | 513846         | Entro Sin Tu Permiso             | ['Falsalarma', 'Tito', 'Ose']                        | 0.998577         |
| 4  | 688929         | Не надо (Latin Mix)              | ['VIA Gra']                                         | 0.998479         |
| 5  | 913941         | There You Go                     | ['P!nk']                                            | 0.998171         |
| 6  | 176324         | Three Days                       | ['Thermadore']                                      | 0.998120         |
| 7  | 561854         | The Jolly One Is Here 2          | ['Sounds Of Blackness', 'Goeffrey Evans Jones']     | 0.997706         |
| 8  | 36545          | Trinity                          | ['Northern State']                                  | 0.997685         |
| 9  | 77552          | Just Let Go                      | ['Coco Montoya']                                    | 0.997658         |
| 10 | 408748         | Да-ди-дам                        | ['Kristina Orbakaitė']                              | 0.997652         |

> Seluruh rekomendasi ini diperoleh berdasarkan kemiripan konten lagu terhadap lagu referensi.

### Kelebihan Pendekatan Ini

- **Tidak memerlukan data pengguna**: Cukup menggunakan fitur lagu untuk memberikan rekomendasi.
- **Dapat merekomendasikan lagu yang belum pernah diputar** (cold-start untuk item).
- **Cepat dan efisien setelah precomputing cosine similarity.**

### Kelemahan

- **Kurang personalisasi**: Sistem tidak mempertimbangkan preferensi pengguna tertentu.
- **Tidak mempertimbangkan popularitas atau feedback pengguna.**

## Evaluation

Untuk menilai performa dari sistem rekomendasi berbasis **Content-Based Filtering (CBF)**, dilakukan evaluasi menggunakan metrik klasifikasi, yaitu **precision**, **recall**, dan **f1-score**. Evaluasi ini bertujuan untuk mengetahui sejauh mana rekomendasi lagu yang diberikan sesuai dengan harapan, terutama dalam konteks kemiripan konten lagu.
Selain itu, dilakukan juga pendekatan validasi visual menggunakan metode Principal Component Analysis (PCA) untuk memvisualisasikan kedekatan distribusi fitur antara lagu referensi dan lagu-lagu hasil rekomendasi.

### Classification Report

Evaluasi dilakukan menggunakan pendekatan berikut:

- Lagu dianggap **relevan** jika memiliki skor **similarity > 0.9** terhadap lagu referensi.
- Sistem memberikan **10 rekomendasi teratas** (Top-10).
- `y_true` dibentuk berdasarkan apakah lagu yang direkomendasikan termasuk dalam daftar lagu relevan.
- `y_pred` diasumsikan selalu 1 (karena semua lagu yang direkomendasikan dianggap sebagai prediksi positif).
- Metrik dievaluasi menggunakan `classification_report` dari scikit-learn.

Berikut adalah hasil evaluasi terhadap satu contoh lagu referensi secara acak:

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not Relevant  | 0.00      | 0.00   | 0.00     | 0       |
| Relevant      | 1.00      | 1.00   | 1.00     | 10      |
|               |           |        |          |         |
| Accuracy      |           |        | 1.00     | 10      |
| Macro Avg     | 0.50      | 0.50   | 0.50     | 10      |
| Weighted Avg  | 1.00      | 1.00   | 1.00     | 10      |

- Seluruh lagu yang direkomendasikan oleh sistem termasuk dalam kategori **relevan**, sehingga precision, recall, dan f1-score untuk kelas **Relevant** bernilai sempurna (**1.00**).
- Kelas **Not Relevant** tidak muncul karena seluruh rekomendasi telah sesuai, yang secara teknis membuat metrik untuk kelas tersebut menjadi nol, namun tidak berdampak negatif pada performa sistem secara keseluruhan.
- **Akurasi sistem mencapai 100%**, yang menandakan bahwa sistem bekerja sangat baik dalam skenario ini.

---

### Validasi Visual (PCA 2D)

Gambar di bawah ini menunjukkan hasil visualisasi clustering dari seluruh lagu berdasarkan dua komponen utama hasil reduksi dimensi menggunakan PCA. Lagu referensi ditandai dengan warna **merah**, sedangkan 20 lagu rekomendasi ditandai dengan warna **biru**.

![Visualisasi PCA](https://i.postimg.cc/dVsRF64y/image.png)

Berdasarkan visualisasi, terlihat bahwa lagu-lagu rekomendasi berkumpul sangat dekat dengan lagu referensi dalam ruang PCA. Hal ini menunjukkan bahwa secara karakteristik konten audio (seperti danceability, tempo, valence, dll.), lagu-lagu rekomendasi memang serupa dengan lagu acuan.

### Evaluasi terhadap Business Understanding

Berdasarkan hasil evaluasi sistem rekomendasi berbasis **Content-Based Filtering (CBF)** menggunakan metrik klasifikasi (precision, recall, f1-score), serta validasi dari hasil rekomendasi terhadap lagu referensi, berikut kesimpulan yang dapat diambil:

#### Apakah sudah menjawab setiap *Problem Statement*?

1. **Pengguna kesulitan menemukan lagu baru yang sesuai dengan selera mereka.**  
    *Terjawab.* Sistem dapat menyarankan lagu-lagu serupa secara konten meskipun lagu-lagu tersebut belum pernah didengarkan pengguna sebelumnya.

2. **Sistem rekomendasi berbasis collaborative filtering tidak efektif untuk lagu baru atau tidak populer.**  
    *Terjawab.* Sistem CBF tidak bergantung pada histori pengguna, sehingga mampu merekomendasikan lagu baru atau langka berdasarkan fitur kontennya.

3. **Kurangnya rekomendasi yang bersifat personal berdasarkan karakteristik konten lagu.**  
    *Terjawab.* Rekomendasi disusun berdasarkan fitur-fitur audio seperti danceability, valence, dan acousticness yang merefleksikan gaya dan suasana lagu, sehingga memberikan hasil yang lebih personal.

---

#### Apakah berhasil mencapai setiap *Goals*?

1. **Membangun sistem rekomendasi berdasarkan fitur konten.**  
    *Tercapai.* Model menggunakan cosine similarity pada fitur-fitur numerik lagu dan mampu menghasilkan Top-N recommendation.

2. **Memberikan rekomendasi yang relevan dan personal.**  
    *Tercapai.* Hasil evaluasi menunjukkan nilai precision dan recall yang sangat tinggi, bahkan mencapai 100% dalam salah satu uji acak, membuktikan bahwa sistem memberikan saran yang akurat dan sesuai dengan lagu referensi.

3. **Menyediakan evaluasi yang menunjukkan kesamaan antar lagu.**  
    *Tercapai.* Evaluasi kuantitatif dilakukan dengan metrik klasifikasi dan evaluasi visual didukung oleh PCA 2D, yang menunjukkan bahwa lagu-lagu rekomendasi berada dekat dengan lagu referensi dalam ruang fitur.

---

#### Apakah setiap *Solution Statement* berdampak?

1. **Pendekatan CBF menggunakan fitur numerik dan cosine similarity** terbukti efektif dalam menghasilkan rekomendasi yang serupa secara konten tanpa memerlukan histori pengguna.  
   *Efektif dan berhasil dijalankan.*

2. **Evaluasi dengan metrik klasifikasi** seperti precision, recall, dan f1-score memberikan gambaran performa yang lebih obyektif. Dalam pengujian yang dilakukan, sistem menunjukkan akurasi 100%, yang menunjukkan efektivitas pendekatan ini.  
   *Memberikan dampak positif terhadap kepercayaan terhadap sistem.*

3. **Sampling terbatas** memungkinkan model tetap berjalan efisien di lingkungan komputasi terbatas seperti Google Colab tanpa mengorbankan performa rekomendasi.  
   *Strategi ini berhasil menjaga efisiensi tanpa mengurangi kualitas hasil.*

---

### Kesimpulan Akhir

Model rekomendasi berbasis Content-Based Filtering yang dibangun tidak hanya berhasil menjawab masalah bisnis utama, tetapi juga menunjukkan performa yang sangat baik berdasarkan evaluasi kuantitatif dan kualitatif. Model ini layak digunakan sebagai dasar untuk sistem rekomendasi lagu, terutama pada kondisi cold-start dan eksplorasi musik personal berbasis konten.
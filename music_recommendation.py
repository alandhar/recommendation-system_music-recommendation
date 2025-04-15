# -*- coding: utf-8 -*-
"""music_recommendation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zHnCoMVYivvRmSun87LKY3f87eIO0-xK

# Import Library & Setup Dataset
"""

from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/MyDrive/Dicoding/5. Machine Learning Terapan'

!mkdir -p ~/.kaggle
!cp '{path}/kaggle.json' ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d rodolfofigueroa/spotify-12m-songs

!unzip -o '/content/spotify-12m-songs.zip' -d '/content'

"""1. Menghubungkan Google Drive agar dapat mengambil file `kaggle.json`.
2. Menyalin `kaggle.json` ke folder `.kaggle` dan mengatur izinnya.
3. Mengunduh dataset Spotify dari Kaggle.
4. Mengekstrak file zip dataset ke direktori `/content`.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

"""- `pandas`, `numpy`: untuk manipulasi dan analisis data.
- `matplotlib`, `seaborn`: untuk visualisasi grafik dan distribusi data.
- `LabelEncoder`, `MinMaxScaler`: untuk encoding dan normalisasi fitur numerik.
- `cosine_similarity`: untuk menghitung kemiripan antar lagu.
- `PCA`: untuk reduksi dimensi dan visualisasi clustering lagu.

## Load Dataset
"""

df = pd.read_csv("/content/tracks_features.csv")
df.head()

"""Membaca dataset Spotify dari file `tracks_features.csv` dan menampilkan 5 baris pertama untuk melihat struktur awal data.

# Data Understanding

## Data Assessing
"""

df.shape

"""Menampilkan ukuran dataset, yaitu terdiri dari 1.204.025 baris dan 24 kolom yang berisi informasi fitur dari lagu-lagu di Spotify

"""

df.info()

"""Menampilkan informasi struktur dataset, termasuk nama kolom, jumlah nilai tidak null di setiap kolom, serta tipe data masing-masing kolom. Informasi ini berguna untuk mengetahui apakah terdapat missing value dan tipe data yang perlu diproses lebih lanjut."""

df.isnull().sum()

"""Menghitung jumlah nilai kosong (missing value) pada setiap kolom. Ditemukan bahwa hanya kolom `name` dan `album` yang memiliki missing value, masing-masing sebanyak 3 dan 11 entri. Kolom lainnya lengkap."""

df = df.dropna()

"""Menghapus seluruh baris yang memiliki nilai kosong."""

df.describe()

"""Menampilkan statistik deskriptif dari fitur numerik dalam dataset, seperti nilai minimum, maksimum, mean, median, dan standar deviasi.

## EDA (Exploratory Data Analysis )

### Distribusi Audio Features
"""

audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']

plt.figure(figsize=(15, 8))
for i, col in enumerate(audio_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[col], kde=True, bins=40)
    plt.title(col)
plt.tight_layout()
plt.show()

"""Visualisasi distribusi lima fitur audio utama (`danceability`, `energy`, `valence`, `acousticness`, dan `instrumentalness`) menggunakan histogram.

- `danceability` menunjukkan distribusi mendekati normal dengan puncak di sekitar 0.5–0.6.
- `energy` memiliki distribusi yang cukup merata dengan sedikit puncak di nilai rendah dan tinggi.
- `valence` (tingkat kebahagiaan lagu) cenderung menurun seiring peningkatan nilainya.
- `acousticness` menunjukkan dua konsentrasi ekstrem: sangat rendah dan sangat tinggi.
- `instrumentalness` sangat terpusat pada nilai rendah, dengan sedikit distribusi di nilai tinggi, menandakan mayoritas lagu memiliki sedikit elemen instrumental.

### Korelasi antar Audio Features
"""

plt.figure(figsize=(10, 8))
sns.heatmap(df[audio_features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Audio Features")
plt.show()

"""Matriks korelasi antar fitur audio menunjukkan hubungan linear antara fitur-fitur numerik lagu.

- `danceability`, `energy`, dan `valence` saling berkorelasi positif, artinya lagu yang enerjik cenderung memiliki kesan ceria dan mudah untuk menari.
- `acousticness` memiliki korelasi negatif yang kuat dengan `energy` (-0.80), menandakan bahwa lagu akustik biasanya memiliki energi rendah.
- `instrumentalness` memiliki korelasi lemah dengan fitur lainnya, menunjukkan bahwa kehadiran instrumen tidak terlalu dipengaruhi oleh aspek danceability atau valence.

### Distribusi Tahun dan Durasi Lagu
"""

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year'] = df['release_date'].dt.year
df['duration_min'] = df['duration_ms'] / 60000

valid_df = df[df['year'].notna() & df['duration_min'].notna()]

"""- Mengonversi kolom `release_date` ke format datetime, lalu mengekstrak tahunnya ke kolom baru `year`, karena kolom `year` sebelumnya terdapat terdapat nilai invalid.
- Menghitung durasi lagu dalam satuan menit dari kolom `duration_ms`.
- Data yang memiliki nilai kosong pada `year` atau `duration_min` disaring dan disimpan dalam `valid_df` untuk memastikan hanya data valid yang digunakan dalam pemodelan.
"""

plt.figure(figsize=(15, 6))
sns.histplot(
    data=valid_df,
    x='year',
    y='duration_min',
    bins=100,
)
plt.title('Distribusi Lagu Berdasarkan Tahun & Durasi')
plt.xlabel('Tahun')
plt.ylabel('Durasi (menit)')
plt.tight_layout()
plt.show()

"""Visualisasi distribusi lagu berdasarkan tahun rilis dan durasi lagu dalam satuan menit. Mayoritas lagu dengan durasi wajar (sekitar 2–5 menit) lebih banyak diproduksi setelah tahun 2000.

### Distribusi Kolom Explicit
"""

plt.figure(figsize=(10, 6))
sns.countplot(data=valid_df, x='explicit')
plt.title('Distribusi Lagu Eksplisit')
plt.xlabel('Explicit')
plt.ylabel('Jumlah Lagu')
plt.xticks([0, 1], ['Tidak', 'Ya'])
plt.tight_layout()
plt.show()

"""Visualisasi distribusi lagu berdasarkan status eksplisit. Sebagian besar lagu dalam dataset tidak mengandung konten eksplisit, dengan hanya sebagian kecil yang ditandai sebagai eksplisit.

> *Note* : "**explicit**" merujuk pada lagu yang mengandung konten dewasa atau sensitif.

# Data Preparation

### Encoding Kolom track_id
"""

track_encoder = LabelEncoder()
valid_df['track_encoded'] = track_encoder.fit_transform(valid_df['id'])

"""Melakukan encoding pada kolom `id` lagu menjadi nilai numerik dengan `LabelEncoder`, lalu disimpan ke kolom baru `track_encoded`. Bertujuan untuk mengubah ID lagu yang berupa string menjadi angka agar bisa digunakan dalam perhitungan similarity dan visualisasi.

### Feature Selection
"""

valid_df = valid_df.sample(n=10000, random_state=42).reset_index(drop=True)

content_features = valid_df[[
    'danceability', 'energy', 'valence',
    'acousticness', 'instrumentalness', 'tempo',
    'duration_min', 'year'
]]

"""Mengambil sampel acak sebanyak 10.000 lagu dari dataset untuk mengurangi beban komputasi saat perhitungan similarity, karena sebelumnya terjadi crash runtime pada google collab. Kemudian, memilih fitur-fitur konten utama dari lagu (seperti karakteristik audio dan metadata numerik) yang akan digunakan dalam model Content-Based Filtering.

### Normalisasi Fitur
"""

scaler = MinMaxScaler()
content_features_scaled = scaler.fit_transform(content_features)

"""Melakukan normalisasi terhadap fitur konten menggunakan `MinMaxScaler` agar seluruh nilai berada dalam rentang 0 hingga 1. Agar semua fitur memiliki kontribusi yang seimbang dalam perhitungan kemiripan antar lagu.

# Modeling : Content-Based Filtering

## Hitung Similarity Antar Lagu
"""

content_similarity = cosine_similarity(content_features_scaled)

similarity_df = pd.DataFrame(
    content_similarity,
    index=valid_df['track_encoded'],
    columns=valid_df['track_encoded']
)

"""Menghitung kemiripan antar lagu berdasarkan fitur kontennya menggunakan cosine similarity, lalu menyimpannya dalam bentuk DataFrame. Nilai similarity ini digunakan untuk merekomendasikan lagu-lagu yang memiliki karakteristik serupa.

## Rekomendasi Lagu Serupa
"""

def recommend_similar_tracks(track_id, top_n=10):
    if track_id not in similarity_df.index:
        return pd.DataFrame()

    sim_scores = similarity_df.loc[track_id]

    sim_scores = sim_scores.sort_values(ascending=False).iloc[1:top_n+1]

    similar_ids = sim_scores.index.tolist()

    similar_tracks = valid_df[valid_df['track_encoded'].isin(similar_ids)][['track_encoded', 'name', 'artists']].copy()

    similar_tracks['similarity_score'] = similar_tracks['track_encoded'].map(sim_scores)

    similar_tracks = similar_tracks.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)

    return similar_tracks

"""Membuat fungsi `recommend_similar_tracks()` untuk merekomendasikan lagu-lagu yang mirip dengan lagu input berdasarkan nilai cosine similarity.

- Menerima `track_id` sebagai input dan jumlah hasil `top_n` yang diinginkan.
- Mengambil nilai similarity terhadap lagu lain, mengurutkannya, dan memilih lagu paling mirip (kecuali dirinya sendiri).
- Mengembalikan DataFrame berisi informasi lagu-lagu yang direkomendasikan beserta skor kemiripannya.

"""

random_track = valid_df.sample(n=1).iloc[0]
example_track = random_track['track_encoded']

print("Lagu referensi:")
valid_df[valid_df['track_encoded'] == example_track][['name', 'artists']]

"""Memilih satu lagu secara acak dari dataset sebagai lagu referensi untuk rekomendasi. Lagu yang dipilih akan digunakan sebagai input untuk mencari lagu-lagu dengan karakteristik konten yang paling mirip."""

recommendations = recommend_similar_tracks(example_track, top_n=20)
print("Rekomendasi lagu serupa:")
recommendations

"""Mengambil 20 lagu yang paling mirip dengan lagu referensi berdasarkan hasil perhitungan cosine similarity, lalu menampilkannya sebagai rekomendasi. Setiap lagu yang direkomendasikan disertai dengan skor kemiripannya.

# Evaluation
"""

pca = PCA(n_components=2)
pca_result = pca.fit_transform(content_features_scaled)

valid_df['pca_1'] = pca_result[:, 0]
valid_df['pca_2'] = pca_result[:, 1]

"""Menggunakan PCA (Principal Component Analysis) untuk mereduksi dimensi fitur konten lagu menjadi dua komponen utama. Hasil transformasi disimpan dalam kolom `pca_1` dan `pca_2`, yang akan digunakan untuk visualisasi distribusi dan klasterisasi lagu dalam 2 dimensi."""

recommendations = recommend_similar_tracks(example_track, top_n=20)
selected_ids = recommendations['track_encoded'].tolist() + [example_track]

valid_df['type'] = 'Lainnya'
valid_df.loc[valid_df['track_encoded'] == example_track, 'type'] = 'Referensi'
valid_df.loc[valid_df['track_encoded'].isin(recommendations['track_encoded']), 'type'] = 'Rekomendasi'

plt.figure(figsize=(10, 6))
colors = {'Lainnya': 'lightgray', 'Referensi': 'red', 'Rekomendasi': 'blue'}

for label, group in valid_df.groupby('type'):
    plt.scatter(group['pca_1'], group['pca_2'],
                label=label, alpha=0.6, s=20 if label == 'Lainnya' else 80,
                color=colors[label])

plt.legend()
plt.title("Visualisasi Clustering Lagu (PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.show()

"""Visualisasi hasil PCA untuk menunjukkan posisi lagu referensi dan lagu-lagu yang direkomendasikan dalam ruang 2 dimensi berdasarkan fitur kontennya.

- Lagu referensi ditampilkan dengan warna merah.
- Lagu-lagu hasil rekomendasi ditampilkan dengan warna biru.
- Lagu lainnya ditampilkan dengan warna abu-abu.

Distribusi ini menunjukkan bahwa lagu-lagu yang direkomendasikan berada dalam cluster yang sama atau dekat dengan lagu referensi, yang menjadi indikasi bahwa sistem rekomendasi berhasil mengidentifikasi kemiripan konten.
"""


# =============================================================================
# KODE PYTHON UNTUK KLUSTERISASI SPBU KE TBBM (VERSI DIPERBAIKI)
# =============================================================================
#
# Deskripsi:
# Skrip ini mengelompokkan SPBU dari file data Anda ke TBBM terdekat,
# menghasilkan visualisasi dengan legenda yang jelas, dan menyimpan hasilnya
# ke dalam file CSV baru.
#
# Library yang dibutuhkan (jika belum install):
# pip install pandas numpy scikit-learn matplotlib folium

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.metrics import pairwise_distances_argmin_min

print("Memulai proses klusterisasi SPBU ke TBBM...")

# --- Bagian 1: MEMUAT DATA ASLI ANDA ---
try:
    df_tbbm = pd.read_csv('tbbm1.csv', encoding='utf-8-sig', sep=';')
    df_spbu = pd.read_csv('spbu1.csv', encoding='utf-8-sig', sep=';')
    print(f"Data berhasil dimuat: {len(df_tbbm)} TBBM dan {len(df_spbu)} SPBU.")
    print("\nKolom data TBBM:", df_tbbm.columns.tolist())
    print("Kolom data SPBU:", df_spbu.columns.tolist())

except FileNotFoundError:
    print("ERROR: File tidak ditemukan!")
    print("Pastikan file 'tbbm1.csv' dan 'spbu1.csv' berada di folder yang sama dengan skrip Python ini.")
    exit()
except Exception as e:
    print(f"Terjadi error saat memuat data: {e}")
    exit()


# --- Bagian 2: PROSES KLUSTERISASI ---
try:
    coords_spbu = df_spbu[['Latitude', 'Longitude']].values
    coords_tbbm = df_tbbm[['Latitude', 'Longitude']].values
except KeyError:
    print("\nERROR: Nama kolom 'Latitude' atau 'Longitude' tidak ditemukan!")
    print("Pastikan nama kolom di file CSV Anda sudah benar dan sesuai (perhatikan huruf besar/kecil).")
    exit()

closest_tbbm_indices, distances = pairwise_distances_argmin_min(coords_spbu, coords_tbbm)

df_spbu['Kluster_TBBM_Index'] = closest_tbbm_indices
df_spbu['Nama_TBBM_Terdekat'] = df_tbbm['Nama'].iloc[closest_tbbm_indices].values
df_spbu['Jarak_ke_TBBM'] = distances

print("\nProses klusterisasi selesai. Setiap SPBU telah ditugaskan ke TBBM terdekat.")
print("\nContoh hasil klusterisasi:")
print(df_spbu.head())


# --- Bagian 3 (BARU): MENYIMPAN HASIL KE CSV ---
# Permintaan Anda untuk menyimpan hasil ke file CSV baru diimplementasikan di sini.
try:
    nama_file_hasil = 'hasil_klusterisasi_spbu.csv'
    # Menyimpan DataFrame yang sudah berisi hasil klusterisasi
    df_spbu.to_csv(nama_file_hasil, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nSUKSES: Hasil klusterisasi lengkap telah disimpan ke file: '{nama_file_hasil}'")
except Exception as e:
    print(f"\nERROR: Gagal menyimpan file CSV. Penyebab: {e}")


# --- Bagian 4: VISUALISASI HASIL (DENGAN LEGENDA JELAS) ---

# A. Visualisasi menggunakan Matplotlib (Scatter Plot)
print("\nMembuat visualisasi scatter plot dengan legenda yang jelas...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 12))

# PERUBAHAN: Membuat legenda yang lebih informatif dengan memetakan warna ke nama TBBM.
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
tbbm_names = df_tbbm['Nama'].unique()
color_map = {name: colors[i % len(colors)] for i, name in enumerate(tbbm_names)}

# Plot setiap kluster secara terpisah untuk membuat legenda
for name, color in color_map.items():
    cluster_data = df_spbu[df_spbu['Nama_TBBM_Terdekat'] == name]
    if not cluster_data.empty: # Hanya plot jika ada SPBU di kluster ini
        ax.scatter(cluster_data['Longitude'], cluster_data['Latitude'], c=color, label=f'Kluster {name}', alpha=0.7, s=50)

# Plot TBBM sebagai bintang besar berwarna hitam
ax.scatter(df_tbbm['Longitude'], df_tbbm['Latitude'], c='black', marker='*', s=500, label='Lokasi TBBM', edgecolors='white', zorder=5)

for i, row in df_tbbm.iterrows():
    ax.text(row['Longitude'] + 0.01, row['Latitude'], row['Nama'], fontsize=12, fontweight='bold')

ax.set_title('Peta Klusterisasi SPBU berdasarkan TBBM Terdekat (Data Asli)', fontsize=16)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.legend(title='Legenda', loc='upper left')
ax.grid(True)

nama_file_plot = 'plot_kluster_spbu_asli_dengan_legenda.png'
plt.savefig(nama_file_plot)
print(f"Plot berhasil disimpan sebagai '{nama_file_plot}'.")


# B. Visualisasi menggunakan Folium (Peta Interaktif)
# (Tidak ada perubahan di bagian ini, sudah informatif)
print("Membuat peta interaktif HTML...")
map_center = [df_spbu['Latitude'].mean(), df_spbu['Longitude'].mean()]
peta_kluster = folium.Map(location=map_center, zoom_start=9, tiles='CartoDB positron')

map_colors = ['blue', 'orange', 'green', 'red', 'purple']
tbbm_color_map_folium = {name: map_colors[i % len(map_colors)] for i, name in enumerate(tbbm_names)}

for i, row in df_tbbm.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"<strong>{row['Nama']}</strong>",
        tooltip="TBBM (Centroid)",
        icon=folium.Icon(color='black', icon='industry', prefix='fa')
    ).add_to(peta_kluster)

for i, row in df_spbu.iterrows():
    cluster_color = tbbm_color_map_folium[row['Nama_TBBM_Terdekat']]
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=cluster_color,
        fill=True,
        fill_color=cluster_color,
        fill_opacity=0.8,
        popup=f"<strong>{row['Nama']}</strong><br>Kluster: {row['Nama_TBBM_Terdekat']}",
        tooltip=row['Nama']
    ).add_to(peta_kluster)

nama_file_peta = 'peta_kluster_spbu_asli.html'
peta_kluster.save(nama_file_peta)
print(f"Peta interaktif berhasil disimpan sebagai '{nama_file_peta}'. Buka file ini di browser Anda.")

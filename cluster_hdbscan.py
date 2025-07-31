# =============================================================================
# KODE PYTHON UNTUK KLUSTERISASI MIKRO DENGAN HDBSCAN (VERSI 5.1 - PERBAIKAN TQDM)
# =============================================================================
#
# Deskripsi:
# Skrip ini melakukan optimasi parameter 'min_cluster_size' untuk HDBSCAN
# menggunakan Genetic Algorithm (GA) untuk setiap daerah pelayanan.
#
# PERUBAHAN (VERSI 5.1):
# - Memperbaiki implementasi TQDM dengan menghapus perintah 'print()' yang
#   mengganggu render loading bar.
# - Menambahkan saran untuk menjalankan skrip di terminal standar.
#
# Library yang dibutuhkan (jika belum install):
# pip install pandas hdbscan matplotlib numpy geneticalgorithm tqdm

import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import os
from tqdm import tqdm # <-- IMPORT LIBRARY UNTUK PROGRESS BAR
import warnings

# Menonaktifkan peringatan yang tidak relevan dari library lain
warnings.filterwarnings("ignore", category=FutureWarning)

print("Memulai proses optimasi dan klusterisasi mikro dengan GA + HDBSCAN (v5.1)...")
print("Strategi Optimasi: Mencari 'min_cluster_size' yang meminimalkan jumlah outlier (noise).")
print("CATATAN: Untuk tampilan loading bar terbaik, jalankan skrip ini dari terminal (CMD/PowerShell/Terminal).")

# --- Bagian 1: MEMUAT DATA ---
try:
    df_hasil_awal = pd.read_csv('hasil_klusterisasi_spbu.csv', sep=';')
    df_tbbm = pd.read_csv('tbbm1.csv', sep=';')
    print("Data hasil klusterisasi dan data TBBM berhasil dimuat.")
except FileNotFoundError as e:
    print(f"ERROR: File tidak ditemukan! Pastikan file '{e.filename}' ada di direktori yang sama.")
    exit()

# Membuat folder untuk menyimpan output jika belum ada
output_folder = 'hasil_klusterisasi_mikro'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' berhasil dibuat untuk menyimpan hasil.")

# --- Bagian 2: PERSIAPAN DATA DAN FUNGSI ---
df_hasil_final = pd.DataFrame()

# --- Bagian 3: OPTIMASI DENGAN GENETIC ALGORITHM ---

# Loop untuk setiap daerah yang unik, sekarang dengan progress bar TQDM
# tqdm akan secara otomatis membuat bilah kemajuan untuk loop ini
list_daerah = df_hasil_awal['Nama_TBBM_Terdekat'].unique()
for daerah in tqdm(list_daerah, desc="Memproses Semua Daerah", unit="daerah"):
    
    # PERBAIKAN: Perintah print() di bawah ini dihapus karena akan
    # merusak tampilan loading bar TQDM di setiap iterasi.
    # Informasi daerah mana yang sedang diproses sudah ditangani oleh TQDM.
    # print(f"\n{'='*20} Memproses Daerah: {daerah} {'='*20}")
    
    # Filter data untuk daerah yang sedang diproses
    df_daerah = df_hasil_awal[df_hasil_awal['Nama_TBBM_Terdekat'] == daerah].copy()
    coords = df_daerah[['Latitude', 'Longitude']].values
    
    # Jika jumlah SPBU terlalu sedikit, lewati optimasi
    if len(df_daerah) < 3:
        df_daerah['Sub_Kluster_ID'] = 0 # Tetapkan semua sebagai satu kluster
        df_hasil_final = pd.concat([df_hasil_final, df_daerah], ignore_index=True)
        continue

    # Fungsi fitness yang akan diminimalkan oleh GA
    def f(X):
        min_size = int(np.round(X[0]))
        if min_size < 2: min_size = 2
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, metric='haversine', gen_min_span_tree=True)
        clusterer.fit(np.radians(coords))
        noise_count = np.sum(clusterer.labels_ == -1)
        return noise_count

    # Batas untuk min_cluster_size
    max_possible_size = max(2, len(df_daerah) // 2)
    varbound = np.array([[2, max_possible_size]])
    
    # Pengaturan untuk Genetic Algorithm
    algorithm_param = {
        'max_num_iteration': 10000,
        'population_size': 25,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type':'uniform',
        'max_iteration_without_improv': 1000
    }

    # Inisialisasi model GA
    model = ga(function=f, dimension=1, variable_type='int', 
               variable_boundaries=varbound, 
               algorithm_parameters=algorithm_param,
               progress_bar=False) # <-- Progress bar bawaan GA sudah benar dinonaktifkan
    
    model.run()
    
    # Ambil hasil terbaik dari GA
    optimal_min_size = int(model.best_variable[0])
    
    # --- Bagian 4: KLUSTERISASI FINAL & VISUALISASI ---
    final_clusterer = hdbscan.HDBSCAN(min_cluster_size=optimal_min_size, metric='haversine', gen_min_span_tree=True)
    final_clusterer.fit(np.radians(coords))
    
    # Tambahkan hasil sub-kluster ke DataFrame
    df_daerah['Sub_Kluster_ID'] = final_clusterer.labels_
    df_hasil_final = pd.concat([df_hasil_final, df_daerah], ignore_index=True)

    # Visualisasi hasil
    try:
        tbbm_info = df_tbbm[df_tbbm['Nama'] == daerah].iloc[0]
    except IndexError:
        tbbm_info = None

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))
    unique_labels = set(final_clusterer.labels_)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            label_text = 'Outlier / Noise'
            col = 'black'
        else:
            label_text = f'Sub-Kluster {k}'
        
        class_member_mask = (final_clusterer.labels_ == k)
        xy = df_daerah[['Longitude', 'Latitude']][class_member_mask]
        ax.scatter(xy['Longitude'], xy['Latitude'], c=[col], s=60, label=label_text, alpha=0.9, edgecolors='w', linewidth=0.5)

    if tbbm_info is not None:
        ax.scatter(tbbm_info['Longitude'], tbbm_info['Latitude'], c='red', marker='*', s=800, label='Lokasi TBBM', edgecolors='black', zorder=10)
        ax.text(tbbm_info['Longitude'], tbbm_info['Latitude'] - 0.015, daerah, fontsize=12, fontweight='bold', ha='center')

    ax.set_title(f'Klusterisasi Optimal HDBSCAN untuk Daerah {daerah}\n(min_cluster_size = {optimal_min_size} ditemukan oleh GA)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True)
    
    # Simpan plot ke file
    nama_file_plot = os.path.join(output_folder, f'plot_kluster_mikro_{daerah.replace(" ", "_")}.png')
    plt.savefig(nama_file_plot, dpi=300, bbox_inches='tight')
    plt.close(fig)

# --- Bagian 5: MENYIMPAN HASIL AKHIR ---
nama_file_output = os.path.join(output_folder, 'hasil_klusterisasi_mikro_final.csv')
df_hasil_final.to_csv(nama_file_output, sep=';', index=False)

print(f"\n{'='*20} PROSES SELESAI {'='*20}")
print(f"Semua daerah telah diproses.")
print(f"Hasil klusterisasi mikro lengkap telah disimpan di: '{nama_file_output}'")

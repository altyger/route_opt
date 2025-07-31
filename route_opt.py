# =============================================================================
# KODE PYTHON UNTUK PENENTUAN RUTE OPTIMAL DENGAN KONSEP SPBU GERBANG
# VERSI 6.0 (MODIFIKASI MENGGUNAKAN SPBU GERBANG)
# =============================================================================
#
# Deskripsi:
# Skrip ini mengimplementasikan strategi rute dua tingkat yang lebih efisien.
# 1. Titik Masuk: Mengganti centroid dengan "SPBU Gerbang", yaitu SPBU di
#    dalam kluster yang jaraknya paling dekat ke TBBM.
# 2. Rute Utama: Rute optimal dari TBBM mengunjungi "SPBU Gerbang" setiap kluster.
# 3. Rute Cabang: Rute optimal dimulai dari "SPBU Gerbang" untuk mengunjungi
#    SPBU lain di dalam kluster yang sama.
#
# Metode:
# - Klusterisasi Mikro: HDBSCAN.
# - Optimasi Rute: Google OR-Tools.
# - Perhitungan Jarak: Rumus Haversine.
#
# Library yang dibutuhkan:
# pip install pandas hdbscan matplotlib numpy scikit-learn ortools

import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

print("Memulai proses penentuan rute optimal dengan konsep SPBU Gerbang (v6.0)...")

# --- Bagian 1: FUNGSI-FUNGSI PEMBANTU ---

def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

def create_distance_matrix(coords):
    n_points = len(coords)
    dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                dist_matrix[i, j] = haversine_distance(
                    coords[i][1], coords[i][0],
                    coords[j][1], coords[j][0]
                )
    return dist_matrix

# --- BARU: Fungsi untuk mencari SPBU Gerbang ---
def find_entry_point_spbu(df_cluster, tbbm_coords):
    """
    Mencari SPBU dalam sebuah kluster yang paling dekat dengan TBBM.
    Mengembalikan informasi (koordinat dan indeks) dari SPBU tersebut.
    """
    min_dist = float('inf')
    entry_point_info = None
    
    for index, row in df_cluster.iterrows():
        spbu_coords = [row['Latitude'], row['Longitude']]
        dist = haversine_distance(tbbm_coords[1], tbbm_coords[0], spbu_coords[1], spbu_coords[0])
        if dist < min_dist:
            min_dist = dist
            entry_point_info = {
                'coords': spbu_coords,
                'original_index': index # Menyimpan index asli dari DataFrame
            }
    return entry_point_info


def solve_tsp_with_ortools(dist_matrix):
    """
    Menyelesaikan TSP menggunakan Google OR-Tools.
    Input adalah matriks jarak (dalam kilometer).
    Output: rute optimal (urutan indeks) dan total jarak.
    """
    num_locations = dist_matrix.shape[0]

    if num_locations < 2:
        return [], 0.0
    if num_locations == 2:
        return [1], dist_matrix[0, 1] * 2

    manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        min_dist_km = solution.ObjectiveValue() / 1000.0
        index = routing.Start(0)
        route_indices = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:
                route_indices.append(node_index)
            index = solution.Value(routing.NextVar(index))
        return route_indices, min_dist_km
    else:
        return None, 0.0

# --- Bagian 2: MEMUAT DATA DAN VALIDASI ---
try:
    # Menggunakan hasil dari klusterisasi mikro
    df_hasil_awal = pd.read_csv('hasil_klusterisasi_mikro/hasil_klusterisasi_mikro_final.csv', sep=';')
    df_tbbm = pd.read_csv('tbbm1.csv', sep=';')
    print("Data hasil klusterisasi mikro dan data TBBM berhasil dimuat.")
except FileNotFoundError:
    print("ERROR: File 'hasil_klusterisasi_mikro_final.csv' tidak ditemukan!")
    print("Pastikan Anda sudah menjalankan skrip 'cluster_hdbscan.py' terlebih dahulu.")
    exit()

for df in [df_hasil_awal, df_tbbm]:
    for col in ['Latitude', 'Longitude']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
print("Validasi data selesai.")

# --- Bagian 3: PROSES UTAMA DENGAN STRATEGI SPBU GERBANG ---

daerah_pelayanan_unik = df_hasil_awal['Nama_TBBM_Terdekat'].unique()
for daerah in daerah_pelayanan_unik:
    print(f"\n--- Memproses Daerah Pelayanan: {daerah} ---")

    df_daerah = df_hasil_awal[df_hasil_awal['Nama_TBBM_Terdekat'] == daerah].copy()
    # Mengganti nama kolom agar konsisten
    df_daerah.rename(columns={'Sub_Kluster_ID': 'Sub_Kluster'}, inplace=True)
    
    # Handle outlier: setiap outlier dianggap sebagai kluster tersendiri
    outlier_mask = df_daerah['Sub_Kluster'] == -1
    df_daerah.loc[outlier_mask, 'Sub_Kluster'] = [f"Outlier_{i}" for i in df_daerah.index[outlier_mask]]

    if len(df_daerah) < 1:
        print(f"Daerah '{daerah}' tidak memiliki SPBU, proses dilewati.")
        continue

    tbbm_info = df_tbbm[df_tbbm['Nama'] == daerah].iloc[0]
    tbbm_coords = [tbbm_info['Latitude'], tbbm_info['Longitude']]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 14))

    # --- DIUBAH: Mencari SPBU Gerbang, bukan Centroid ---
    sub_clusters = df_daerah['Sub_Kluster'].unique()
    cluster_entry_points = {}
    for cluster_id in sub_clusters:
        df_sub_cluster = df_daerah[df_daerah['Sub_Kluster'] == cluster_id]
        entry_point_info = find_entry_point_spbu(df_sub_cluster, tbbm_coords)
        cluster_entry_points[cluster_id] = entry_point_info
    
    print(f"Ditemukan {len(sub_clusters)} sub-kluster. SPBU Gerbang untuk setiap kluster telah ditentukan.")

    # 4. Cari Rute Utama (TBBM -> SPBU Gerbang -> TBBM)
    print("Mencari Rute Utama (TBBM ke SPBU Gerbang) dengan OR-Tools...")
    entry_point_coords_list = [info['coords'] for info in cluster_entry_points.values()]
    main_route_nodes = [tbbm_coords] + entry_point_coords_list
    main_dist_matrix = create_distance_matrix(main_route_nodes)
    
    best_main_route_indices, main_total_dist = solve_tsp_with_ortools(main_dist_matrix)
    
    if best_main_route_indices is None:
        print("Gagal menemukan rute utama.")
        continue

    sorted_entry_points = [entry_point_coords_list[i-1] for i in best_main_route_indices]
    main_route_coords = [tbbm_coords] + sorted_entry_points + [tbbm_coords]

    main_lats, main_lons = zip(*main_route_coords)
    ax.plot(main_lons, main_lats, color='black', linewidth=3, linestyle='-', marker='s', markersize=8, label=f'Rute Utama ({main_total_dist:.2f} km)', zorder=5)
    print(f"Rute Utama ditemukan dengan total jarak: {main_total_dist:.2f} km")

    # 5. Cari dan Plot Rute Cabang untuk setiap kluster
    total_branch_dist = 0
    cluster_colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(sub_clusters)))

    for i, cluster_id in enumerate(sub_clusters):
        df_sub_cluster = df_daerah[df_daerah['Sub_Kluster'] == cluster_id]
        entry_point_info = cluster_entry_points[cluster_id]
        entry_point_coords = entry_point_info['coords']
        color = cluster_colors[i]
        
        # --- DIUBAH: Rute cabang dari SPBU Gerbang ke SPBU lain ---
        # Buat daftar SPBU lain di dalam kluster (tidak termasuk gerbangnya)
        other_spbu_df = df_sub_cluster[df_sub_cluster.index != entry_point_info['original_index']]
        
        # Plot semua SPBU di kluster
        ax.scatter(df_sub_cluster['Longitude'], df_sub_cluster['Latitude'], c=[color], s=60, alpha=0.9, edgecolors='black')
        # Tandai SPBU Gerbang secara khusus
        ax.scatter(entry_point_coords[1], entry_point_coords[0], c=[color], marker='P', s=250, label=f'Gerbang Kluster {cluster_id}', edgecolors='black', zorder=6)
        
        if not other_spbu_df.empty:
            print(f"  Mencari Rute Cabang untuk Kluster {cluster_id}...")
            other_spbu_coords = other_spbu_df[['Latitude', 'Longitude']].values.tolist()
            branch_nodes = [entry_point_coords] + other_spbu_coords
            branch_dist_matrix = create_distance_matrix(branch_nodes)
            
            best_branch_route_indices, branch_dist = solve_tsp_with_ortools(branch_dist_matrix)
            
            if best_branch_route_indices:
                sorted_spbu_coords = [other_spbu_coords[j-1] for j in best_branch_route_indices]
                branch_route_coords = [entry_point_coords] + sorted_spbu_coords + [entry_point_coords]
                
                branch_lats, branch_lons = zip(*branch_route_coords)
                ax.plot(branch_lons, branch_lats, color=color, linewidth=1.5, linestyle='--', marker='o', markersize=4)
                total_branch_dist += branch_dist
                print(f"  Rute Cabang Kluster {cluster_id} ditemukan. Jarak: {branch_dist:.2f} km")

    # 6. Finalisasi Plot
    ax.scatter(tbbm_info['Longitude'], tbbm_info['Latitude'], c='red', marker='*', s=800, label='Lokasi TBBM', edgecolors='black', zorder=10)
    ax.text(tbbm_info['Longitude'], tbbm_info['Latitude'] - 0.02, daerah, fontsize=14, fontweight='bold', ha='center')
    
    total_dist_combined = main_total_dist + total_branch_dist
    ax.set_title(f'Rute Distribusi Optimal untuk Daerah {daerah} (Model SPBU Gerbang)\n(Total Jarak: {total_dist_combined:.2f} km)', fontsize=20, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    nama_file_plot = f'plot_rute_gerbang_{daerah.replace(" ", "_")}.png'
    plt.savefig(nama_file_plot, bbox_inches='tight')
    print(f"\nPlot rute optimal untuk daerah '{daerah}' berhasil disimpan sebagai '{nama_file_plot}'.")
    plt.close(fig)

print("\n\nProses penentuan rute untuk semua daerah pelayanan telah selesai.")


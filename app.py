import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Membaca Data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)
    
# Atur Tema
st.markdown(
    """
    <style>
    :root {
        --primary-color: #f0f2f6;
        --background-color: #ffffff;
        --secondary-background-color: #D0D0D0;
        --text-color: #262730;
        --font: sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
#st.set_page_config(page_title="Clustering App For Leads Auto2000 Kramat Jati", page_icon=":bar_chart:", layout="wide")
st.image("logoa2000.png", width=250)
st.title("Clustering App For Leads Auto2000 Kramat Jati")
st.write("Upload your CSV file to perform clustering")

# Melakukan Upload Data
file = st.file_uploader("Select a CSV file", type=["csv"])

if file is not None:
    # Load data
    df = load_data(file)
    if 'action' in df.columns:
        df.drop(columns=['Action'], inplace=True)
    # Display data
    #st.write("Upload Data:")
    #st.write(df)

    # Kolom untuk melakukan clustering
    cols = ['Keinginan membeli', 'Kesiapan pembayaran fee', 'Kapan dapat ditemui secara langsung', 'Frekuensi penggunaan']

   # Memisahkan kolom kolom
    additional_cols = ['Phone', 'Model', 'Product Desc.', 'Anggaran pembelian', 'Metode pembayaran']
    if all(col in df.columns for col in cols + additional_cols + ['Customer Name', 'Reference to']):
        
        # Membuat clustering data frame
        clustering_data = df[cols + additional_cols + ['Customer Name', 'Reference to']]

        # Encode label data kategori
        mappings = {
            'Keinginan membeli': {'0-1 Bulan': 2, '1-3 Bulan': 1, '3-6 Bulan': 0},
            'Kesiapan pembayaran fee': {'Minggu ini': 2, 'Bulan ini': 1, 'Belum menentukan': 0},
            'Kapan dapat ditemui secara langsung': {'1-2 Minggu': 2, '1 Bulan': 1, 'Belum menentukan': 0},
            'Frekuseni penggunaan': {'Setiap hari': 2, 'Di akhir pekan': 1, 'Sesekali': 0}
        }

        for col, mapping in mappings.items():
            if col in clustering_data.columns:
                clustering_data[col] = clustering_data[col].map(mapping)

        # Normalisasi Data
        scaler = MinMaxScaler()
        clustering_data[cols] = scaler.fit_transform(clustering_data[cols])

        # Melakukan clustering
        kmeans = KMeans(
            n_clusters=3,
            init=np.array([[0.0, 0.0, 0.0, 0.0], [0.4, 0.4, 0.4, 0.4], [0.8, 0.8, 0.8, 0.8]]),
            algorithm='elkan',
            random_state=64,
            n_init=1,
            max_iter=100
        )
        clustering_data['cluster'] = kmeans.fit_predict(clustering_data[cols])

        # Mapping cluster
        cluster_mapping = {0: 'Low', 1: 'Mid', 2: 'Hot'}
        clustering_data['cluster_label'] = clustering_data['cluster'].map(cluster_mapping)

        # Menghitung silhouette score
        #silhouette_avg = silhouette_score(clustering_data[cols], clustering_data['cluster'])
        #st.write("Silhouette score clustering:", silhouette_avg)
        
        # Menghitung setiap centroid cluster
        #centroids = kmeans.cluster_centers_
        #st.write("Centroid setiap cluster:")
        #for i, centroid in enumerate(centroids):
        #    st.write(f"Cluster {i}: {centroid}")

        # Menampilkan hasil clustering
        st.write("Hasil clustering:")
        st.write(clustering_data[['Customer Name', 'Reference to', 'cluster', 'cluster_label']])

        # Memanngil kolom kembali
        ordered_cols = ['Customer Name', 'Reference to', 'Phone', 'Model', 'Product Desc.', 'Anggaran pembelian', 'Metode pembayaran', 'cluster', 'cluster_label'] + cols
        clustering_data = clustering_data[ordered_cols]

        # Menampilkan hasil
        #st.write("Data Final Clustering:")
        #st.write(clustering_data)

        # Filter berdasarkan label cluster
        selected_clusters = st.multiselect("Pilih cluster untuk ditampilkan", options=['Low', 'Mid', 'Hot'], default=['Low', 'Mid', 'Hot'])
        filtered_data = clustering_data[clustering_data['cluster_label'].isin(selected_clusters)]

        # Menampilkan data yang difilter
        st.write("Data setelah difilter berdasarkan cluster:")
        st.write(filtered_data)

        # Menghitung jarak Euclidean dari setiap data ke centroid
        #def euclidean_distance(point, centroid):
         #   return np.sqrt(np.sum((point - centroid)**2))

        #clustering_data['distance_to_centroid'] = clustering_data.apply(
          #  lambda row: euclidean_distance(row[cols].values, centroids[row['cluster']]),
         #   axis=1
        #)

        #st.write("Data with Distance to Centroid:")
        #st.write(clustering_data)
        
    else:
        st.write("The uploaded CSV file does not contain all the required columns.")

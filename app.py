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

    # Display data
    st.write("Upload Data:")
    st.write(df)

    # Kolom untuk melakukan clustering
    cols = ['Keinginan memiliki mobil', 'Kesiapan pembayaran booking fee', 'Kapan dapat ditemui secara langsung', 'Frekuseni penggunaan mobil']

   # Memisahkan kolom kolom
    additional_cols = ['Phone', 'Model', 'Product Desc.', 'Anggaran untuk membeli mobil', 'Metode pembayaran yang diinginkan']
    if all(col in df.columns for col in cols + additional_cols + ['Nama Customer', 'Reference To']):
        
        # Membuat clustering data frame
        clustering_data = df[cols + additional_cols + ['Nama Customer', 'Reference To']]

        # Encode label data kategori
        mappings = {
            'Keinginan memiliki mobil': {'0-1 bulan': 2, '1-3 bulan': 1, '3-6 bulan': 0},
            'Kesiapan pembayaran booking fee': {'minggu ini': 2, 'bulan ini': 1, 'belum menentukan': 0},
            'Kapan dapat ditemui secara langsung': {'1-2 minggu': 2, '1 bulan': 1, 'belum menentukan': 0},
            'Frekuseni penggunaan mobil': {'setiap hari': 2, 'diakhir pekan': 1, 'sesekali': 0}
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
        cluster_mapping = {0: 'low', 1: 'mid', 2: 'hot'}
        clustering_data['cluster_label'] = clustering_data['cluster'].map(cluster_mapping)

        # Menghitung silhouette score
        silhouette_avg = silhouette_score(clustering_data[cols], clustering_data['cluster'])
        st.write("Silhouette score clustering:", silhouette_avg)
        
        # Menghitung setiap centroid cluster
        centroids = kmeans.cluster_centers_
        st.write("Centroid setiap cluster:")
        for i, centroid in enumerate(centroids):
            st.write(f"Cluster {i}: {centroid}")

        # Menampilkan hasil clustering
        st.write("Hasil clustering:")
        st.write(clustering_data[['Nama Customer', 'Reference To', 'cluster', 'cluster_label']])

        # Memanngil kolom kembali
        ordered_cols = ['Nama Customer', 'Reference To', 'Phone', 'Model', 'Product Desc.', 'Anggaran untuk membeli mobil', 'Metode pembayaran yang diinginkan', 'cluster', 'cluster_label'] + cols
        clustering_data = clustering_data[ordered_cols]

        # Menampilkan hasil
        st.write("Data Final Clustering:")
        st.write(clustering_data)

        # Menghitung jarak Euclidean dari setiap data ke centroid
        def euclidean_distance(point, centroid):
            return np.sqrt(np.sum((point - centroid)**2))

        clustering_data['distance_to_centroid'] = clustering_data.apply(
            lambda row: euclidean_distance(row[cols].values, centroids[row['cluster']]),
            axis=1
        )

        st.write("Data with Distance to Centroid:")
        st.write(clustering_data)
        
        # Apply PCA to reduce dimensions to 2D
        try:
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(clustering_data[cols])
            clustering_data['pca1'] = pca_components[:, 0]
            clustering_data['pca2'] = pca_components[:, 1]

            # Plotting the clustering result using PCA scatter plot
            st.write("PCA Clustering Visualisasi:")
            fig, ax = plt.subplots()
            scatter = ax.scatter(clustering_data['pca1'], clustering_data['pca2'], c=clustering_data['cluster'])
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.set_title("PCA Clustering Result")

            # Adding legend
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)

            # Display the plot in Streamlit
            st.pyplot(fig)
        except Exception as e:
            st.write("An error occurred during PCA transformation:")
            st.write(e)
    else:
        st.write("The uploaded CSV file does not contain all the required columns.")

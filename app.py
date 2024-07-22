import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Load data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Create a Streamlit app
st.title("Clustering App")
st.write("Upload your CSV file to perform clustering")

# File uploader
file = st.file_uploader("Select a CSV file", type=["csv"])

if file is not None:
    # Load the data
    df = load_data(file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(df)

    # Select the columns to cluster
    cols = st.multiselect("Select columns to cluster", df.columns)

    if len(cols) > 0:
        # Create a clustering data frame
        clustering_data = df[cols + ['Nama Customer', 'Reference To']]

        # Encode categorical data
        mappings = {
            'Keinginan memiliki mobil': {'0-1 bulan': 2, '1-3 bulan': 1, '3-6 bulan': 0},
            'Kesiapan pembayaran booking fee': {'minggu ini': 2, 'bulan ini': 1, 'belum menentukan': 0},
            'Kapan dapat ditemui secara langsung': {'1-2 minggu': 2, '1 bulan': 1, 'belum menentukan': 0},
            'Frekuseni penggunaan mobil': {'setiap hari': 2, 'diakhir pekan': 1, 'sesekali': 0}
        }

        for col, mapping in mappings.items():
            if col in clustering_data.columns:
                clustering_data[col] = clustering_data[col].map(mapping)

        # Normalize the data
        scaler = MinMaxScaler()
        clustering_data[cols] = scaler.fit_transform(clustering_data[cols])

        # Perform clustering
        kmeans = KMeans(n_clusters=3, init=np.array([[0.0, 0.0, 0.0, 0.0], [0.4, 0.4, 0.4, 0.4], [0.8, 0.8, 0.8, 0.8]]))
        n_iter = st.selectbox("Number of iterations (n_iter)", ["1", "10", "30", "50"])
        max_iter = st.selectbox("Maximum number of iterations (max_iter)", [ "16", "32", "64", "128", "256"])
        algorithm = st.selectbox("Algorithm", ["auto", "full", "elkan"])
        clustering_data['cluster'] = kmeans.fit_predict(clustering_data[cols])

        # Calculate the Silhouette score
        silhouette_avg = silhouette_score(clustering_data[cols], clustering_data['cluster'])
        st.write("Silhouette score:", silhouette_avg)

        # Calculate the centroid coordinates for each cluster
        centroids = kmeans.cluster_centers_
        st.write("Centroid coordinates for each cluster:")
        for i, centroid in enumerate(centroids):
            st.write(f"Cluster {i}: {centroid}")

        # Visualize the clusters
        #fig, ax = plt.subplots()
        #ax.scatter(clustering_data[cols[0]], clustering_data[cols[1]], c=clustering_data['cluster'])
        #ax.set_xlabel(cols[0])
        #ax.set_ylabel(cols[1])
        #ax.set_title("Clustering Result")
        #st.pyplot(fig)

        # Plotting the clustering result using scatter plot
        st.write("Clustering Visualization:")
        fig, ax = plt.subplots()
        scatter = ax.scatter(clustering_data[cols[0]], clustering_data[cols[1]], c=clustering_data['cluster'])
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_title("Clustering Result")
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)
        
        # Display the clustering result
        st.write("Clustering result:")
        st.write(clustering_data[['Nama Customer', 'Reference To', 'cluster']])

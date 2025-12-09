# baby_nutrition_cluster.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="BabyNutrition Cluster", layout="wide")

st.title("BabyNutrition Cluster 🍼")
st.write("Aplikasi untuk mengelompokkan makanan bayi berdasarkan kandungan nutrisinya menggunakan KMeans clustering.")

# ======== Baca CSV otomatis dari folder lokal ========
csv_file_path = "standard-nutrition.csv"
csv_file_path = "foods.csv"  # pastikan file ada di folder yang sama dengan app.py

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    st.subheader("Data Makanan Bayi")
    st.dataframe(df.head())

    # Pilih kolom nutrisi yang digunakan untuk clustering
    available_columns = df.columns[2:]  # kecuali kolom index dan Menu
    default_columns = ['Energy (kJ)','Protein (g)','Fat (g)','Carbohydrates (g)']

    # Hanya pakai default yang benar-benar ada di CSV
    default_columns = [col for col in default_columns if col in available_columns]

    nutri_columns = st.multiselect(
        "Pilih kolom nutrisi untuk clustering",
        options=available_columns,
        default=default_columns
)


    if len(nutri_columns) >= 2:
        # Standarisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[nutri_columns])

        # Pilih jumlah cluster
        n_clusters = st.slider("Pilih jumlah cluster (K)", min_value=2, max_value=10, value=3)

        # Buat model KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df['Cluster'] = clusters

        st.subheader("Hasil Clustering")
        st.dataframe(df[['Menu', 'Cluster'] + nutri_columns].head(20))

        # Statistik cluster
        st.subheader("Rata-rata Nutrisi per Cluster")
        st.dataframe(df.groupby('Cluster')[nutri_columns].mean().round(2))
        
        st.subheader("Rekomendasi Makanan Terbaik untuk Bayi")

        # Ambil centroid cluster
        centroids = kmeans.cluster_centers_

        # Hitung skor nutrisi untuk setiap cluster (semakin tinggi semakin baik)
        cluster_scores = centroids.sum(axis=1)

        # Tentukan cluster terbaik (nutrisi paling tinggi)
        best_cluster = cluster_scores.argmax()

        st.write(f"Cluster terbaik berdasarkan nilai nutrisi: **Cluster {best_cluster}**")

        # Filter makanan dalam cluster terbaik
        best_foods = df[df["Cluster"] == best_cluster].copy()

        from sklearn.metrics import pairwise_distances

        # Hitung jarak tiap makanan ke centroid
        distances = pairwise_distances(
            scaler.transform(best_foods[nutri_columns]),
            centroids[best_cluster].reshape(1, -1)
        )

        best_foods["DistanceToCentroid"] = distances

        # Ambil makanan terdekat (rekomendasi utama)
        recommended_food = best_foods.sort_values("DistanceToCentroid").iloc[0]

        st.success(f"### ⭐ Rekomendasi Utama: **{recommended_food['Menu']}**")
        st.write("Makanan ini paling mendekati komposisi nutrisi ideal dalam cluster terbaik.")

        # Detail nutrisi rekomendasi
        st.subheader("Detail Nutrisi Rekomendasi")
        st.dataframe(recommended_food[nutri_columns + ["Cluster", "DistanceToCentroid"]].to_frame().T)

        # Alternatif makanan lain
        st.subheader("Alternatif Makanan Lain dalam Cluster Terbaik")
        st.dataframe(
            best_foods.sort_values("DistanceToCentroid")
                        .head(5)[["Menu", "Cluster", "DistanceToCentroid"] + nutri_columns]
        )

    # Visualisasi cluster (2 fitur pertama)
        st.subheader("Visualisasi Cluster")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(
            x=df[nutri_columns[0]], 
            y=df[nutri_columns[1]], 
            hue=df['Cluster'], 
            palette='Set2', 
            s=100,
            ax=ax
        )
        ax.set_title("Cluster Makanan Bayi")
        ax.set_xlabel(nutri_columns[0])
        ax.set_ylabel(nutri_columns[1])
        st.pyplot(fig)


    else:
        st.warning("Pilih minimal 2 kolom nutrisi untuk clustering.")
else:
    st.error(f"File '{csv_file_path}' tidak ditemukan. Silakan letakkan file CSV di folder project.")




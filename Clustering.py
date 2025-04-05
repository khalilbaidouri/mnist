import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Remplacez 4 par le nombre de cœurs que vous souhaitez utiliser
# Charger les données depuis un fichier CSV
def load_data(file_path):
    data = []
    with open(file_path, encoding="utf-8-sig") as file:
        filereader = csv.reader(file)
        for row in filereader:
            # Convertir chaque élément de la ligne en entier
            row_data = [int(num) for num in row]
            data.append(np.array(row_data))
    return data

# Séparer les features (X) et les labels (Y)
def split_features_and_labels(data):
    X = [row[:-1] for row in data]  # Toutes les colonnes sauf la dernière
    Y = [row[-1] for row in data]   # Dernière colonne comme étiquette
    return X, Y

# Appliquer K-means et évaluer le clustering
def apply_kmeans(X, Y, n_clusters=12):
    # Créer et entraîner le modèle K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Calculer les scores de qualité du clustering
    ami_score = adjusted_mutual_info_score(cluster_labels, Y)  # Adjusted Mutual Information
    ari_score = adjusted_rand_score(cluster_labels, Y)         # Adjusted Rand Index

    # Afficher les scores
    print(f"Adjusted Mutual Information (AMI) Score: {np.round(ami_score * 100, 2)}")
    print(f"Adjusted Rand Index (ARI) Score: {np.round(ari_score * 100, 2)}")

    return kmeans, cluster_labels

# Visualiser les centroïdes des clusters
def visualize_clusters(kmeans, n_clusters):
    fig, axes = plt.subplots(1, n_clusters, figsize=(15, 5))  # Créer une figure avec des sous-graphiques
    for c_i in range(n_clusters):
        # Récupérer et redimensionner le centroïde du cluster
        centroid_image = kmeans.cluster_centers_[c_i].reshape(12, 6)  # Redimensionner en 12x6

        # Afficher l'image du centroïde
        axes[c_i].imshow(centroid_image, cmap='gray')  # Utiliser une colormap en niveaux de gris

        # Supprimer les ticks des axes pour une meilleure lisibilité
        axes[c_i].set_yticks([])
        axes[c_i].set_xticks([])

    # Ajuster l'espacement et afficher la figure
    plt.tight_layout()
    plt.show()

# Fonction principale
def main():
    # Charger les données
    data = load_data("data.csv")

    # Séparer les features et les labels
    X, Y = split_features_and_labels(data)

    # Appliquer K-means et évaluer le clustering
    kmeans, cluster_labels = apply_kmeans(X, Y, n_clusters=12)

    # Visualiser les centroïdes des clusters
    visualize_clusters(kmeans, n_clusters=12)

# Point d'entrée du programme
if __name__ == "__main__":
    main()
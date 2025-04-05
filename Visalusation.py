import csv
import numpy as np
import matplotlib.pyplot as plt
import math

# === Lecture du fichier CSV ===
def lire_csv(nom_fichier):
    images = []
    labels = []
    with open(nom_fichier, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 73:  # 72 pixels + 1 label
                print(f"Ligne ignorée, taille inattendue : {len(row)} valeurs.")
                continue
            pixels = [int(n) for n in row[:-1]]
            label = int(row[-1])
            image = np.array(pixels).reshape((12, 6))
            images.append(image)
            labels.append(label)
    return images, labels

# === Affichage de toutes les images ===
def afficher_toutes_les_images(images, labels):
    nb_images = len(images)
    cols = 10
    rows = math.ceil(nb_images / cols)
    plt.figure(figsize=(cols * 1.5, rows * 1.5))

    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray_r')
        plt.title(str(label))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# === Programme principal ===
def main():
    fichier_csv = "data.csv"
    images, labels = lire_csv(fichier_csv)
    print(f"{len(images)} images chargées depuis {fichier_csv}.")
    afficher_toutes_les_images(images, labels)

if __name__ == "__main__":
    main()

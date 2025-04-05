import csv
import numpy as np
import itertools
import random
from scipy.ndimage import shift, rotate


# === Fonctions de transformations réalistes ===

def translate_image(image, max_shift=2):
    """Translate l'image horizontalement et/ou verticalement."""
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    translated = shift(image, (shift_x, shift_y), cval=0)
    return np.clip(translated, 0, 1)


def rotate_image(image, max_angle=20):
    """Fait pivoter l'image d'un angle aléatoire."""
    angle = random.uniform(-max_angle, max_angle)
    rotated = rotate(image, angle, reshape=False, cval=0)
    return np.clip(rotated, 0, 1)


def flip_image(image):
    """Inverse les pixels de l'image."""
    return 1 - image


def add_random_noise(image, noise_prob=0.1):
    """Ajoute du bruit binaire en inversant des pixels avec une certaine probabilité."""
    mask = np.random.rand(*image.shape) < noise_prob
    noised_image = np.where(mask, 1 - image, image)
    return noised_image


def stretch_image(image, stretch_factor=1.2):
    """Étirer ou comprimer l'image (simple scaling horizontal/vertical)."""
    rows, cols = image.shape
    scaled = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            ni = min(int(i * stretch_factor), rows - 1)
            nj = min(int(j * stretch_factor), cols - 1)
            scaled[ni][nj] = image[i][j]
    return scaled


# === Lecture du fichier CSV ===

def lire_csv(nom_fichier):
    dataset = []
    labels = []
    with open(nom_fichier, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            valeurs = [int(n) for n in row[:-1]]  # 72 valeurs pour l'image
            label = int(row[-1])  # Dernière colonne pour le label
            image = np.array(valeurs).reshape((12, 6))
            dataset.append(image)
            labels.append(label)
    return dataset, labels


# === Générer toutes les variations possibles ===

def generer_variations(images, labels, nb_variations=5):
    variations = []
    variations_labels = []
    transformations = [translate_image, rotate_image, flip_image, add_random_noise, stretch_image]

    for image, label in zip(images, labels):
        for _ in range(nb_variations):
            transformation = random.choice(transformations)
            nouvelle_image = transformation(image)
            variations.append(nouvelle_image)
            variations_labels.append(label)
    return variations, variations_labels


# === Sauvegarde dans un fichier CSV ===

def sauvegarder_csv(images, labels, nom_fichier):
    with open(nom_fichier, 'w', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.writer(fichier_csv)
        for image, label in zip(images, labels):
            ligne = [int(val) for val in image.flatten()] + [label]
            writer.writerow(ligne)
    print(f"{len(images)} variations enregistrées dans {nom_fichier}.")


# === Programme principal ===

def main():
    fichier_entree = "images_ideals.csv"
    fichier_sortie = "data.csv"

    print("Lecture des images...")
    images, labels = lire_csv(fichier_entree)

    print("Génération des variations...")
    variations, variations_labels = generer_variations(images, labels, nb_variations=10)

    print("Sauvegarde des données augmentées...")
    sauvegarder_csv(variations, variations_labels, fichier_sortie)

    print("Processus terminé !")


if __name__ == "__main__":
    main()

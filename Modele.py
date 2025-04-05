import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tkinter as tk
import numpy as np

# 1. Créer un Dataset personnalisé pour charger les données à partir du CSV
class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Charger les données CSV
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Récupérer les pixels et l'étiquette
        image = self.data.iloc[idx, :-1].values  # Toutes les colonnes sauf la dernière
        label = int(self.data.iloc[idx, -1])    # Dernière colonne comme étiquette

        # Convertir les pixels en un tableau NumPy
        image = image.astype('float32').reshape(12, 6)

        # Appliquer les transformations, si spécifié
        if self.transform:
            image = self.transform(image)

        return image, label

# 2. Définir les transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertir en tenseur
])

# 3. Charger les données depuis le fichier CSV
train_dataset = CSVDataset(csv_file='data.csv', transform=transform)

# Modifier la taille du batch à 16
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 4. Définir le modèle
class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        self.fc1 = nn.Linear(12 * 6, 20)  # Entrée de 12x6 pixels
        self.fc2 = nn.Linear(20, 10)      # Couche cachée avec 20 neurones
        self.fc3 = nn.Linear(10, 10)      # Couche de sortie avec 10 neurones
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 12 * 6)  # Aplatir l'image en un vecteur de 12x6
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ImprovedModel()

# 5. Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
# Modifier le taux d'apprentissage à 0.01
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. Fonction pour calculer la précision
def calculate_accuracy(loader):
    model.eval()  # Passer le modèle en mode évaluation
    correct = 0
    total = 0
    with torch.no_grad():  # Désactiver le calcul du gradient pour la validation
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # Obtenir les prédictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total  # Calculer la précision
    return accuracy

# 7. Entraîner le modèle
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0  # Pour stocker la perte moyenne par époque

    # Phase d'entraînement
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculer la perte moyenne pour l'époque
    epoch_loss = running_loss / len(train_loader)

    # Calculer la précision sur les données d'entraînement
    train_accuracy = calculate_accuracy(train_loader)

    print(f"Époque {epoch + 1}/{epochs}, Perte : {epoch_loss:.4f}, Précision : {train_accuracy:.2f}%")

# Sauvegarder le modèle
torch.save(model.state_dict(), "improved_model.pth")
print("Le modèle a été sauvegardé dans 'improved_model.pth'")

# Charger le modèle préalablement entraîné
model = ImprovedModel()
model.load_state_dict(torch.load("improved_model.pth"))
model.eval()  # Passer le modèle en mode évaluation

# Fonction pour dessiner et mettre à jour la grille
def draw(event, canvas, grid, row, col):
    x1 = col * cell_size
    y1 = row * cell_size
    x2 = (col + 1) * cell_size
    y2 = (row + 1) * cell_size
    index = row * cols + col  # Calcul de l'indice dans le vecteur de 72 éléments
    if grid[index] == 0:
        canvas.create_rectangle(x1, y1, x2, y2, fill="black")
        grid[index] = 1
    else:
        canvas.create_rectangle(x1, y1, x2, y2, fill="white")
        grid[index] = 0

# Fonction pour prédire le chiffre basé sur la grille
def predict(grid):
    # Convertir la grille en un tenseur PyTorch
    grid_tensor = torch.tensor(grid, dtype=torch.float32).view(1, -1)  # Mettre la grille sous forme de vecteur
    with torch.no_grad():  # Pas de calcul des gradients pour la prédiction
        outputs = model(grid_tensor)  # Propagation avant
        _, predicted = torch.max(outputs, 1)  # Récupérer l'indice du label prédit
    return predicted.item()

# Fonction pour afficher et prédire le chiffre
def predict_digit():
    prediction = predict(grid)
    result_label.config(text=f"Chiffre prédit : {prediction}")

# Créer la fenêtre principale
root = tk.Tk()
root.title("Grille 12x6 pour Dessiner un Chiffre")

# Dimensions de la grille
rows = 12
cols = 6
cell_size = 40  # Taille des cases

# Créer un vecteur pour stocker l'état de chaque cellule (0 pour blanc, 1 pour noir)
grid = [0] * (rows * cols)

# Créer le canevas pour dessiner
canvas = tk.Canvas(root, width=cols * cell_size, height=rows * cell_size)
canvas.pack()

# Dessiner la grille
for i in range(rows):
    for j in range(cols):
        canvas.create_rectangle(j * cell_size, i * cell_size, (j + 1) * cell_size, (i + 1) * cell_size, outline="black", fill="white")

# Lier l'événement de dessin au clic sur la grille
canvas.bind("<Button-1>", lambda event, canvas=canvas: draw(event, canvas, grid, event.y // cell_size, event.x // cell_size))

# Ajouter un bouton pour prédire le chiffre
predict_button = tk.Button(root, text="Prédire le chiffre", command=predict_digit)
predict_button.pack()

# Label pour afficher la prédiction
result_label = tk.Label(root, text="Chiffre prédit : ")
result_label.pack()

# Lancer la boucle principale de l'application
root.mainloop()
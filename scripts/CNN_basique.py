#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:15:33 2026

@author: frocha
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Simulation d'un mini-dataset d'images
# Batch de 4 images, 1 canal (Niveaux de gris), dimensions 28x28
torch.manual_seed(42)
X = torch.randn(4, 1, 28, 28)
y = torch.tensor([0, 1, 2, 1])  # 3 classes possibles (0, 1, 2)


# 2. Définition de l'architecture du CNN
class BasicCNN(nn.Module):

  def __init__(self):
    super(BasicCNN, self).__init__()

    # Couche de Convolution : 1 canal en entrée, 8 filtres en sortie
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    self.relu = nn.ReLU()
    # Couche de Pooling (réduction de taille) : divise par 2 (28x28 -> 14x14)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Couche linéaire finale (Fully Connected)
    # Après la convolution et le pooling, l'image fait 14x14 avec 8 canaux
    self.fc = nn.Linear(8 * 14 * 14, 3)

  def forward(self, x):
    # Passage dans la convolution -> ReLU -> Pooling
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)

    # Aplatissement (Flatten) pour passer de la matrice 2D au vecteur 1D
    # x.size(0) conserve la taille du batch, -1 aplatit le reste
    x = x.view(x.size(0), -1)

    # Couche de classification finale (logits pour les 3 classes)
    x = self.fc(x)
    return x


# Instanciation du modèle, de la perte et de l'optimiseur
model = BasicCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Boucle d'entraînement ultra-rapide (5 époques)
for epoch in range(5):
  # Forward pass
  outputs = model(X)
  loss = criterion(outputs, y)

  # Backward pass
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  print(f"Époque [{epoch+1}/5] - Perte (Loss): {loss.item():.4f}")

# 4. Prédiction sur une nouvelle image simulée
nouvelle_image = torch.randn(1, 1, 28, 28)
model.eval()
with torch.no_grad():
  logits = model(X[0,:,:,:])
  classe_predite = torch.argmax(logits, dim=1)

print(f"\nClasse prédite pour la nouvelle image : {classe_predite.item()}")
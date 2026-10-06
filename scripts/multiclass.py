#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:16:20 2026

@author: frocha
"""

import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Génération de données synthétiques
# 300 échantillons, 2 features
torch.manual_seed(42)
X = torch.randn(300, 2)

# Attribution de 3 classes (0, 1, 2) selon une règle simple sur les features
y = torch.zeros(300, dtype=torch.long)
y[(X[:, 0] + X[:, 1] > 0.5)] = 1
y[(X[:, 0] - X[:, 1] > 1.0)] = 2


plt.figure(1)
plt.title("données")
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="coolwarm",
    edgecolors="k",
    alpha=0.75
)


# 2. Définition du modèle simple
# Entrée : 2 features -> Couche cachée (ReLU) -> Sortie : 3 classes (logits)
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))

# 3. Fonction de perte et Optimiseur
criterion = nn.CrossEntropyLoss() # SoftMax inclus implicitement
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Boucle d'entraînement (100 époques)
for epoch in range(1000):
  # Forward pass : calcul des prédictions
  outputs = model(X)
  loss = criterion(outputs, y)

  # Backward pass : mise à jour des poids
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 25 == 0:
    print(f"Époque [{epoch+1}/100] - Perte (Loss): {loss.item():.4f}")

# 5. Test de prédiction sur un nouveau point (ex: [0.5, -0.5])
X_test = torch.randn(30, 2)
y_test = torch.zeros(30, dtype=torch.long)
y_test[(X_test[:, 0] + X_test[:, 1] > 0.5)] = 1
y_test[(X_test[:, 0] - X_test[:, 1] > 1.0)] = 2

model.eval()  # Mode évaluation
with torch.no_grad():
  logits = model(X_test)
  y_pred = torch.argmax(logits, dim=1)

plt.figure(2)
plt.title("Données Test")
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap="coolwarm",
    edgecolors="k",
    alpha=0.75
)


plt.figure(3)
plt.title("Predictions")
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    cmap="coolwarm",
    edgecolors="k",
    alpha=0.75
)

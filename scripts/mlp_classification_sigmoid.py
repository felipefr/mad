#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:02:00 2026

@author: frocha
"""

import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================
# 1. Dataset très simple
# =====================================

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.1, 0.2],
    [0.2, 0.1],
    [0.9, 0.8],
    [0.8, 0.9],
], dtype=torch.float32)

# Labels : classification binaire
y = torch.tensor([
    0, 1, 1, 0,
    0, 0, 1, 1
], dtype=torch.float32).reshape(-1, 1)


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


# =====================================
# 2. Définition du MLP
# =====================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()  # Conversion en probabilité
        )

    def forward(self, x):
        return self.network(x)


# =====================================
# 3. Initialisation
# =====================================

model = MLP()

# BCELoss prend des probabilités
criterion = nn.BCELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# =====================================
# 4. Entraînement
# =====================================

epochs = 1000

for epoch in range(epochs):

    # Forward pass
    probabilities = model(X)

    # Calcul de la loss
    loss = criterion(
        probabilities,
        y
    )

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )


# =====================================
# 5. Évaluation
# =====================================

model.eval()

with torch.no_grad():

    # Le modèle renvoie directement
    # des probabilités grâce à Sigmoid
    probabilities = model(X)

    # Classification avec seuil 0.5
    predictions = (
        probabilities >= 0.5
    ).float()

    # Calcul de l'accuracy
    accuracy = (
        predictions == y
    ).float().mean()

    print("\nProbabilités :")
    print(probabilities)

    print("\nPrédictions :")
    print(predictions)

    print(
        f"\nAccuracy : {accuracy.item() * 100:.2f}%"
    )
    
plt.figure(2)
plt.title("prediction")
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=predictions,
    cmap="coolwarm",
    edgecolors="k",
    alpha=0.75
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:49:55 2026

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

# Entrées : 2 features
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


# =====================================
# 2. Définition du MLP
# =====================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.network(x)


# =====================================
# 3. Initialisation
# =====================================

model = MLP()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# =====================================
# 4. Entraînement
# =====================================

epochs = 100

for epoch in range(epochs):

    # Forward pass
    logits = model(X)

    # Calcul de la loss
    loss = criterion(logits, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )


# =====================================
# 5. Évaluation
# =====================================

model.eval()

with torch.no_grad():

    logits = model(X)

    # Transformation en probabilités
    probabilities = torch.sigmoid(logits)

    # Seuil à 0.5
    predictions = (probabilities >= 0.5).float()

    accuracy = (predictions == y).float().mean()

    print("\nProbabilités :")
    print(probabilities)

    print("\nPrédictions :")
    print(predictions)

    print(
        f"\nAccuracy : {accuracy.item() * 100:.2f}%"
    )
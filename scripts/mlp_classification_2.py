#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:53:47 2026

@author: frocha
"""


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# =========================================
# 1. Reproductibilité
# =========================================

np.random.seed(20)
torch.manual_seed(20)


# =========================================
# 2. Création du dataset
# =========================================

X, y = make_moons(
    n_samples=400,
    noise=0.15,
    random_state=42
)

# Séparation train / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Conversion en tenseurs PyTorch
X_train_t = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_t = torch.tensor(
    y_train,
    dtype=torch.float32
).view(-1, 1)

X_test_t = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_t = torch.tensor(
    y_test,
    dtype=torch.float32
).view(-1, 1)


# =========================================
# 3. Visualisation des données
# =========================================

plt.figure(figsize=(7, 5))

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="coolwarm",
    edgecolors="k",
    alpha=0.75
)

plt.title("Dataset : deux classes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, alpha=0.3)

plt.show()


# =========================================
# 4. Définition du MLP
# =========================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# =========================================
# 5. Initialisation
# =========================================

model = MLP()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# =========================================
# 6. Entraînement
# =========================================

epochs = 1000

losses = []

for epoch in range(epochs):

    # Mode entraînement
    model.train()

    # Forward pass
    logits = model(X_train_t)

    # Calcul de la loss
    loss = criterion(
        logits,
        y_train_t
    )

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # Sauvegarde de la loss
    losses.append(loss.item())

    # Affichage
    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


# =========================================
# 7. Évaluation du modèle
# =========================================

model.eval()

with torch.no_grad():

    test_logits = model(X_test_t)

    test_probs = torch.sigmoid(
        test_logits
    )

    test_preds = (
        test_probs >= 0.5
    ).float()

    accuracy = (
        test_preds == y_test_t
    ).float().mean().item()

print(
    f"\nAccuracy test : {accuracy:.2%}"
)


# =========================================
# 8. Visualisation de la loss
# =========================================

plt.figure(figsize=(7, 5))

plt.plot(
    losses
)

plt.title("Évolution de la loss")

plt.xlabel("Epoch")

plt.ylabel("BCE Loss")

plt.grid(True, alpha=0.3)

plt.show()


# =========================================
# 9. Grille pour la frontière de décision
# =========================================

x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5

y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(

    np.linspace(
        x_min,
        x_max,
        300
    ),

    np.linspace(
        y_min,
        y_max,
        300
    )

)

grid = torch.tensor(
    np.c_[
        xx.ravel(),
        yy.ravel()
    ],
    dtype=torch.float32
)


# =========================================
# 10. Prédictions sur la grille
# =========================================
model.eval()

with torch.no_grad():
    zz = torch.sigmoid(
        model(grid)
    ).numpy().reshape(xx.shape)

# =========================================
# 11. Visualisation de la frontière
# =========================================

plt.figure(figsize=(7, 5))

# Régions de décision
plt.contourf(
    xx,
    yy,
    zz,
    levels=50,
    cmap="coolwarm",
    alpha=0.35
)

# Frontière à probabilité 0.5
plt.contour(
    xx,
    yy,
    zz,
    levels=[0.5],
    colors="black",
    linewidths=2
)

# Données de test
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap="coolwarm",
    edgecolors="k"
)

plt.title(
    f"Frontière de décision "
    f"(Accuracy = {accuracy:.2%})"
)

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.grid(True, alpha=0.3)

plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:10:25 2026

@author: frocha
"""
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Charger le jeu de données (3 classes)
iris = load_iris()
X, y = iris.data, iris.target

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



# 2. Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Créer et entraîner le modèle multiclasse
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Faire des prédictions
y_pred = model.predict(X_test)

# 5. Évaluer les performances
print("Précision :", accuracy_score(y_test, y_pred))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))


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
plt.title("prediction")
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    cmap="coolwarm",
    edgecolors="k",
    alpha=0.75
)

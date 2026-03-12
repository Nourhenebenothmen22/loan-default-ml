# Loan Default Prediction - Machine Learning Project

Ce projet est une solution complète de Machine Learning pour prédire les défauts de paiement de prêts. Il inclut un pipeline de traitement de données, l'entraînement de modèles avec suivi MLflow, et une API de prédiction prête pour le déploiement.

## 🚀 Fonctionnalités

- **Pipeline de Machine Learning** : De la donnée brute à la prédiction.
- **Gestion du déséquilibre** : Utilisation de SMOTE pour améliorer la détection des classes minoritaires (cas de défaut).
- **Tracking Expérimental** : Intégration complète avec MLflow pour suivre les paramètres et les métriques.
- **API REST** : Interface FastAPI pour servir le modèle en temps réel.
- **Conteneurisation** : Prêt pour la production avec Docker et Docker Compose.

## 📂 Structure du Projet

```text
loan-default-ml/
├── api/                # Code de l'API FastAPI
│   └── main.py         # Point d'entrée de l'API
├── data/               # Données brutes et transformées
│   ├── raw/
│   └── processed/
├── models/             # Modèles entraînés (.pkl)
├── src/                # Scripts Python (entraînement, évaluation)
│   ├── train_model.py  # Script d'entraînement avec MLflow
│   └── predict.py      # Script utilitaire de prédiction
├── notebooks/          # Exploratory Data Analysis (EDA)
├── Dockerfile          # Configuration de l'image Docker
├── compose.yaml        # Orchestration multi-conteneurs
└── requirements.txt    # Dépendances Python
```

## 🛠️ Installation et Utilisation Locale

### 1. Cloner le projet et créer l'environnement

```powershell
# Créer et activer l'environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Entraîner le modèle

Le script d'entraînement effectue une recherche d'hyperparamètres et enregistre le meilleur modèle localement et dans MLflow :

```powershell
python src/train_model.py
```

### 3. Lancer l'API

```powershell
uvicorn api.main:app --reload
```

L'API sera disponible sur `http://localhost:8000`.

---

## 🐳 Déploiement avec Docker (Recommandé)

La méthode la plus simple pour tout lancer est d'utiliser Docker Compose. Cela démarre à la fois l'API et le serveur MLflow.

```bash
docker compose up --build
```

### Services accessibles :

- **FastAPI** : [http://localhost:8000](http://localhost:8000)
- **MLflow UI** : [http://localhost:5000](http://localhost:5000)

---

## 📡 Utilisation de l'API

### Vérifier si l'API est en ligne

```bash
curl http://localhost:8000/
```

### Prédire un défaut de prêt

Envoyez une liste de caractéristiques (features) correspondant aux entrées du modèle :

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": [0.5, 2.1, 0.0, 1.5, ...]}'
```

---

## 📊 Suivi des Expériences

MLflow est configuré pour enregistrer :

- Les hyperparamètres (n_estimators, max_depth, etc.).
- Les métriques de performance (Accuracy, Recall, Precision, F1-Score, ROC AUC).
- Les graphiques (Matrice de confusion).
- Le modèle sérialisé.

---

## 📝 Auteur

Projet développé dans le cadre d'un mini-projet de Machine Learning.

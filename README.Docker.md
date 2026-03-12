# Documentation Docker - Loan Default Prediction

Ce projet utilise Docker pour simplifier le déploiement de l'API de prédiction et le suivi des expériences avec MLflow.

## Lancement de l'application

Pour démarrer tous les services, exécutez la commande suivante à la racine du projet :

```bash
docker compose up --build
```

## Services disponibles

Une fois les conteneurs lancés, les services suivants sont accessibles :

- **API de prédiction (FastAPI)** : [http://localhost:8000](http://localhost:8000)
- **Interface MLflow (Tracking Server)** : [http://localhost:5000](http://localhost:5000)

## Utilisation de l'API

### Vérification de l'état

```bash
curl http://localhost:8000/
```

### Faire une prédiction

Envoyez une requête POST à l'url `/predict` avec les caractéristiques nécessaires :

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"data": [0.5, 1.0, 2.3, ...]}'
```

## Structure Docker

- **Dockerfile** : Utilise une base Python 3.11-slim, installe les dépendances nécessaires et lance l'API avec Uvicorn.
- **compose.yaml** : Orchestre deux services :
  1. `api` : L'application FastAPI.
  2. `mlflow` : Serveur de suivi pour enregistrer les paramètres et métriques des modèles.

## Persistence

Les volumes Docker sont configurés pour que les éléments suivants soient persistés même après l'arrêt des conteneurs :

- Le répertoire `models/` (contenant les fichiers `.pkl`).
- Le répertoire `mlruns/` et le fichier `mlflow.db` pour le suivi MLflow.

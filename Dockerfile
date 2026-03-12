# Utilisation d'une image Python légère et stable
ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION} as base

# Métadonnées de l'image
LABEL maintainer="ML Team"
LABEL description="API de prédiction de défaut de prêt (Loan Default Prediction)"

# Désactive la génération des fichiers .pyc par Python
ENV PYTHONDONTWRITEBYTECODE=1

# Assure que les logs Python sont envoyés directement au terminal sans être mis en tampon
ENV PYTHONUNBUFFERED=1

# Répertoire de travail dans le conteneur
WORKDIR /app

# Création d'un utilisateur non-privilégié pour des raisons de sécurité
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Installation des dépendances système si nécessaire (ex: build-essential pour certains packages ML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
# Utilisation d'un cache pour accélérer les builds ultérieurs
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Passage à l'utilisateur non-privilégié
USER appuser

# Copie du code source dans le conteneur
# On copie tout le projet pour avoir accès aux modèles et au code source
COPY . .

# Exposition du port utilisé par FastAPI
EXPOSE 8000

# Commande pour lancer l'API avec Uvicorn
# On utilise --host 0.0.0.0 pour permettre l'accès externe
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

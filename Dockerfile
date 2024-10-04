# Utiliser une image de base Python
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans l'image Docker
COPY requirements.txt /app/

# Installer les dépendances définies dans le fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers et dossiers du projet dans le conteneur
COPY . /app/

# Exposer le port que Streamlit va utiliser
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# 1. Utiliser une image officielle Python
FROM python:3.10-slim

# 2. Installer des dépendances système (pour pdf, tesseract, etc. si besoin)
RUN apt-get update && \
    apt-get install -y gcc poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# 3. Créer et utiliser un répertoire de travail
WORKDIR /app

# 4. Copier les fichiers requirements et le code
COPY requirements.txt .
COPY . .

# 5. Installer les dépendances Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 6. Exposer le port (optionnel, pour uvicorn)
EXPOSE 8000

# 7. Commande de lancement de l’API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

WORKDIR /app

# Les dépendances d'abord : cette couche est mise en cache tant que requirements.txt ne change pas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src

# Permet d'importer le paquet `provisionnement` depuis /app/src
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "provisionnement.api:app", "--host", "0.0.0.0", "--port", "8000"]

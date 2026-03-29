FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data

# Wait for Ollama to be ready, then ingest, then run the bot entrypoint.
CMD ["sh", "-c", "python scripts/wait_for_ollama.py && python rag/ingester.py && python app.py"]

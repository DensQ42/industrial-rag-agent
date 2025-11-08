FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY api/ ./api/
COPY src/langchain_RAG.py ./src/langchain_RAG.py
COPY data/processed/final_chunks.json ./data/processed/final_chunks.json

COPY .env ./

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import requests
import json

client = QdrantClient("http://localhost:6333")

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

EMBEDDING_DIMENSIONALITY = 512

# for model in TextEmbedding.list_supported_models():
#     if model["dim"] == EMBEDDING_DIMENSIONALITY:
#         print(json.dumps(model, indent=2))

model_handle = "jinaai/jina-embeddings-v2-small-en"

collection_name = "zoomcamp_rag"


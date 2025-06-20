from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import requests
import json

client = QdrantClient("http://localhost:6333")

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

EMBEDDING_DIMENSIONALITY = 512


model_handle = "jinaai/jina-embeddings-v2-small-en"

collection_name = "zoomcamp_rag"

points = []
id = 0

for course in documents_raw:
    for doc in course['documents']:

        point = models.PointStruct(
            id=id,
            vector=models.Document(text=doc['text'], model=model_handle), #embed text locally with "jinaai/jina-embeddings-v2-small-en" from FastEmbed
            payload={
                "text": doc['text'],
                "section": doc['section'],
                "course": course['course']
            } #save all needed metadata fields
        )
        points.append(point)

        id += 1

client.upsert(
    collection_name=collection_name,
    points=points
)
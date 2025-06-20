from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import requests
import json

client = QdrantClient("http://localhost:6333")

EMBEDDING_DIMENSIONALITY = 512

collection_name = "zoomcamp_rag"

# create collection with parameters
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size = EMBEDDING_DIMENSIONALITY,
        distance = models.Distance.COSINE
    )
)
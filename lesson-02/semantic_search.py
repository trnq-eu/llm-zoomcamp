from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import requests
import json
import random

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

# running similarity search
def search(query, limit=1):
    results = client.query_points(
        collection_name = collection_name,
        query = models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle 
        ),
        limit = limit,
        with_payload = True
    )

    return results

# pick a random question from data
course = random.choice(documents_raw)
course_piece = random.choice(course['documents'])
# print(json.dumps(course_piece, indent=2))

# answer
result = search(course_piece['question'])
# 

# print(f"Question:\n{course_piece['question']}\n")
# print("Top Retrieved Answer:\n{}\n".format(result.points[0].payload['text']))
# print("Original Answer:\n{}".format(course_piece['text']))

# let’s search the answer to a question that wasn’t in the initial dataset.
print(search("What if I submit homeworks late?").points[0].payload['text'])

# turn on indexing of payload fields.
client.create_payload_index(
    collection_name = collection_name,
    field_name = "course",
    field_schema = "keyword"
)

def search_in_course(query, course="mlops-zoomcamp", limit=1):
    results = client.query_points(
        collection_name = collection_name,
        query = models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle 
        ),
        query_filter = models.Filter(
            must = [
                models.FieldCondition(
                    key = "course",
                    match = models.MatchValue(value=course)
                )
            ]
        ),
        limit = limit,
        with_payload = True
    )

    return results

print(search_in_course("What if I submit homeworks late?", "mlops-zoomcamp").points[0].payload['text'])

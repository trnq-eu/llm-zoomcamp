from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import requests
import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from tqdm import tqdm
import os
import json
import io
import logging
import numpy as np  # Import numpy

# --- Setup logging ---
logging.basicConfig(
    filename='evaluation_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def hit_rate(relevance_total):
    return sum(any(line) for line in relevance_total) / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank, relevant in enumerate(line):
            if relevant:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth, desc="Evaluating"):
        doc_id = q['document']
        try:
            results = search_function(query=q['question'], course=q['course'])
            relevance = [(d.get('id') == doc_id) for d in results]
        except ValueError as e:
            logging.error(f"Evaluation failed for query: {q['question']}. Error: {str(e)}")
            relevance = [False] * 5  # Assume no relevance, adjust based on your needs
        relevance_total.append(relevance)
    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

client = QdrantClient(host="localhost", port=6333)
CACHE_DIR = 'cached_data'
os.makedirs(CACHE_DIR, exist_ok=True)

url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'

docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
local_docs_path = os.path.join(CACHE_DIR, 'documents-with-ids.json')

def load_documents():
    if os.path.exists(local_docs_path):
        try:
            with open(local_docs_path, 'r') as f:
                documents = json.load(f)
            if not documents:
                raise ValueError("Cached documents file is empty or invalid.")
            return documents
        except Exception:
            os.remove(local_docs_path)
    response = requests.get(docs_url, timeout=10)
    response.raise_for_status()
    documents = response.json()
    with open(local_docs_path, 'w') as f:
        json.dump(documents, f, indent=4)
    return documents

documents = load_documents()

ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
local_ground_truth_path = os.path.join(CACHE_DIR, 'ground-truth-data.csv')

def load_ground_truth():
    if os.path.exists(local_ground_truth_path):
        try:
            df = pd.read_csv(local_ground_truth_path)
            if df.empty:
                raise ValueError("Empty ground truth")
            return df
        except Exception:
            os.remove(local_ground_truth_path)
    response = requests.get(ground_truth_url, timeout=10)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df.to_csv(local_ground_truth_path, index=False)
    return df

df_ground_truth = load_ground_truth()
ground_truth = df_ground_truth.to_dict(orient='records')

texts = []
docs_for_embedding = []

for doc in documents:
    t = doc['question'] + ' ' + doc['text']
    texts.append(t)
    docs_for_embedding.append(doc)

EMBEDDING_DIMENSIONALITY = 512
model_handle = "jinaai/jina-embeddings-v2-small-en"
collection_name = "homework"
embedding_model = TextEmbedding(model_handle)

def setup_collection():
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        if collection_info.points_count != len(documents):
            logging.info(f"Recreating collection '{collection_name}'...")
            client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=EMBEDDING_DIMENSIONALITY, distance=models.Distance.COSINE),
            )
            return True
        else:
            logging.info(f"Collection '{collection_name}' is up-to-date.")
            return False
    except Exception as e:
        if "Not found" in str(e):
            logging.info(f"Collection '{collection_name}' not found. Creating it...")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=EMBEDDING_DIMENSIONALITY, distance=models.Distance.COSINE),
            )
            return True
        raise

needs_upsert = setup_collection()

if needs_upsert:
    logging.info(f"Generating embeddings for {len(texts)} documents...")
    BATCH_SIZE = 32
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i:i + BATCH_SIZE]
        embeddings.extend(embedding_model.embed(batch))

    points = []
    for i, text_embedding in enumerate(embeddings):
        doc = docs_for_embedding[i]
        point = models.PointStruct(
    id=i,  # usare un intero, obbligatorio per Qdrant
    vector=text_embedding.tolist(),
    payload={
        "text": texts[i],
        "section": doc.get("section", ""),
        "course": doc.get("course", ""),
        "question": doc.get("question", ""),
        "original_id": doc.get("id", "")  # salviamo comunque l'id originale per il confronto dopo
    }
    )
        points.append(point)

    logging.info(f"Upserting {len(points)} points to Qdrant...")
    client.upsert(collection_name=collection_name, points=points, wait=True)
    logging.info("Upsert complete.")
else:
    logging.info("No upsert needed.")

def qdrant_search(query, course, limit=5):
    try:
        # Embed the query
        # embedding_model.embed returns a generator, so convert it to a list first
        query_embeddings_list = list(embedding_model.embed([query]))

        # Now you can safely access the first (and only) embedding
        query_embedding_np = query_embeddings_list[0]
        query_vector = query_embedding_np.tolist()

    except Exception as e:
        logging.error(f"Embedding error for query '{query}': {e}")
        raise  # Re-raise the original exception

    qdrant_filter = None
    if course:
        qdrant_filter = models.Filter(
            must=[models.FieldCondition(key="course", match=models.MatchValue(value=course))]
        )

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True
    )

    results = []
    for point in search_result.points:  # Iterate through search_result.points
        # Access payload through point.payload
        result = {
            'text': point.payload.get('text', ''),
            'section': point.payload.get('section', ''),
            'course': point.payload.get('course', ''),
            'question': point.payload.get('question', ''),
            'id': point.payload.get('original_id', point.id)  # Use original_id
        }
        results.append(result)

    return results


# --- Evaluation ---
print("Valutazione della ricerca vettoriale:")
eval_results_vector = evaluate(ground_truth, qdrant_search)
print(f"Hit rate: {eval_results_vector['hit_rate']:.4f}")
print(f"MRR: {eval_results_vector['mrr']:.4f}")

# Save results
with open("evaluation_results.csv", "w") as f:
    f.write("metric,value\n")
    f.write(f"hit_rate,{eval_results_vector['hit_rate']:.4f}\n")
    f.write(f"mrr,{eval_results_vector['mrr']:.4f}\n")

logging.info("Evaluation completed and results saved.")
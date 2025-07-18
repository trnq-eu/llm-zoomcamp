from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import requests
import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding # Make sure TextEmbedding is imported
import os
import json
import io

def hit_rate(relevance_total):
    """
    Calculates the hit rate.
    Hit rate is the proportion of queries for which the correct document
    was found in the search results.
    """
    cnt = 0
    for line in relevance_total:
        if True in line: # Check if at least one relevant document was found
            cnt = cnt + 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    """
    Calculates the Mean Reciprocal Rank (MRR).
    MRR is a statistic for evaluating any process that produces a list of
    possible responses to a sample of queries, ordered by probability of correctness.
    """
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True: # If a relevant document is found at this rank
                total_score = total_score + 1 / (rank + 1) # Add reciprocal rank to total score
                break # Move to the next query once the first relevant document is found
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    """
    Evaluates a search function against a ground truth dataset.

    Args:
        ground_truth (list): A list of dictionaries, where each dictionary
                             represents a query and its expected document ID.
        search_function (function): A function that takes a query dictionary
                                    and returns a list of search result dictionaries.

    Returns:
        dict: A dictionary containing the hit rate and MRR.
    """
    relevance_total = []

    # Iterate through each query in the ground truth
    for q in tqdm(ground_truth):
        doc_id = q['document'] # The expected document ID for the current query
        results = search_function(query=q['question'], course=q['course']) # Get search results from the provided function
        # Create a boolean list indicating relevance for each result.
        # Added a check for 'id' key to prevent KeyError, ensuring robustness.
        relevance = [(d['id'] == doc_id if 'id' in d else False) for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

client = QdrantClient(host="localhost", port=6333) # Using host and port for clarity
CACHE_DIR = 'cached_data'
os.makedirs(CACHE_DIR, exist_ok=True) # Ensure cache directory exists

# Define the URL prefix for the data
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'

# --- Fetch documents with IDs ---
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
local_docs_path = os.path.join(CACHE_DIR, 'documents-with-ids.json')

if os.path.exists(local_docs_path):
    print(f"Loading documents from local cache: {local_docs_path}")
    try:
        with open(local_docs_path, 'r') as f:
            documents = json.load(f)
        if not documents:
            raise ValueError("Cached documents file is empty or invalid.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading cached documents: {e}. Redownloading...")
        if os.path.exists(local_docs_path): # Check again in case of race condition
            os.remove(local_docs_path)
        print(f"Downloading documents from: {docs_url}")
        response = requests.get(docs_url)
        response.raise_for_status()
        documents = response.json()
        with open(local_docs_path, 'w') as f:
            json.dump(documents, f, indent=4)
        print(f"Documents saved to local cache: {local_docs_path}")
else:
    print(f"Downloading documents from: {docs_url}")
    response = requests.get(docs_url)
    response.raise_for_status()
    documents = response.json()
    with open(local_docs_path, 'w') as f:
        json.dump(documents, f, indent=4)
    print(f"Documents saved to local cache: {local_docs_path}")

# --- Fetch ground truth data ---
ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
local_ground_truth_path = os.path.join(CACHE_DIR, 'ground-truth-data.csv')

if os.path.exists(local_ground_truth_path):
    print(f"Loading ground truth from local cache: {local_ground_truth_path}")
    try:
        df_ground_truth = pd.read_csv(local_ground_truth_path)
        if df_ground_truth.empty:
            raise ValueError("Cached ground truth file is empty or invalid.")
    except (pd.errors.EmptyDataError, ValueError) as e:
        print(f"Error loading cached ground truth: {e}. Redownloading...")
        if os.path.exists(local_ground_truth_path):
            os.remove(local_ground_truth_path)
        print(f"Downloading ground truth from: {ground_truth_url}")
        response = requests.get(ground_truth_url)
        response.raise_for_status()
        df_ground_truth = pd.read_csv(io.StringIO(response.text))
        df_ground_truth.to_csv(local_ground_truth_path, index=False)
        print(f"Ground truth saved to local cache: {local_ground_truth_path}")
else:
    print(f"Downloading ground truth from: {ground_truth_url}")
    response = requests.get(ground_truth_url)
    response.raise_for_status()
    df_ground_truth = pd.read_csv(io.StringIO(response.text))
    df_ground_truth.to_csv(local_ground_truth_path, index=False)
    print(f"Ground truth saved to local cache: {local_ground_truth_path}")


ground_truth = df_ground_truth.to_dict(orient='records')

texts = []
# Store original document data alongside text for payload creation
docs_for_embedding = [] 

EMBEDDING_DIMENSIONALITY = 512 # This should match your chosen model's output dimension
model_handle = "jinaai/jina-embeddings-v2-small-en" # Jina v2 typically has 512 dimensions

collection_name = "homework"

# Initialize the TextEmbedding model once
embedding_model = TextEmbedding(model_handle)


# Prepare texts and their corresponding original document data
for doc in documents:
    t = doc['question'] + ' ' + doc['text']
    texts.append(t)
    docs_for_embedding.append(doc) # Keep the original doc to get section/course


# --- Qdrant Collection Management and Upsert ---
try:
    # Get collection info to check if it exists and its properties
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists with {collection_info.points_count} points.")

    # Check if the collection needs to be recreated (e.g., if the number of points doesn't match)
    # Or if you want to force a re-upload for new data/model changes
    if collection_info.points_count != len(documents):
        print(f"Collection '{collection_name}' has {collection_info.points_count} points, but {len(documents)} documents are available. Recreating collection and upserting.")
        client.delete_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=EMBEDDING_DIMENSIONALITY, distance=models.Distance.COSINE),
        )
        needs_upsert = True
    else:
        print(f"Collection '{collection_name}' is up-to-date. Skipping upsert.")
        needs_upsert = False

except Exception as e:
    # If the collection does not exist, QdrantClient will raise an exception
    # specifically, if "Not found" is in the error message
    if "Not found" in str(e):
        print(f"Collection '{collection_name}' not found. Creating collection...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=EMBEDDING_DIMENSIONALITY, distance=models.Distance.COSINE),
        )
        needs_upsert = True
    else:
        # Re-raise other unexpected exceptions
        raise

if needs_upsert:
    print(f"Generating embeddings and preparing {len(texts)} points for upsert...")
    points = []
    # Generate embeddings in a batch for efficiency if possible, or individually
    # FastEmbed's TextEmbedding can take a list of texts
    embeddings = embedding_model.embed(texts) # This will return a list of numpy arrays

    for i, text_embedding in enumerate(embeddings):
        # Retrieve the original document associated with this text
        original_doc = docs_for_embedding[i] 

        point = models.PointStruct(
            id=i, # Use index as ID, or original_doc['id'] if available and unique
            vector=text_embedding.tolist(), # Convert numpy array to list for Qdrant
            payload={
                "text": texts[i], # The full text that was embedded
                "section": original_doc['section'],
                "course": original_doc['course'],
                "question": original_doc['question'] # You might want to save the question too
            }
        )
        points.append(point)

    print(f"Upserting {len(points)} points to Qdrant...")
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True # Wait for the operation to complete
    )
    print("Upsert complete.")
else:
    print("No upsert needed.")

# You can now define your Qdrant search function
def qdrant_search(query, course, limit=5):
    # Embed the query
    query_embedding = embedding_model.embed(query)[0].tolist() # Embed the query

    # Define filter if course is provided
    qdrant_filter = None
    if course:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="course",
                    match=models.MatchValue(value=course)
                )
            ]
        )

    # Perform the search
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True # Retrieve the payload (metadata)
    )

    results = []
    for hit in search_result:
        results.append({
            'text': hit.payload['text'],
            'section': hit.payload['section'],
            'course': hit.payload['course'],
            'question': hit.payload.get('question') # Use .get() in case 'question' isn't always there
        })
    return results

# Example of how to evaluate (uncomment when your evaluation logic is ready)
# --- Valuta la funzione di ricerca vettoriale ---
print("Valutazione della ricerca vettoriale:")
# Assume vector_search_function is defined or use qdrant_search
# For proper evaluation, `evaluate` function expects a search function that takes (query, course)
eval_results_vector = evaluate(ground_truth, qdrant_search)

# Stampa i risultati
print(f"Hit rate per la ricerca vettoriale: {eval_results_vector['hit_rate']:.4f}")
print(f"MRR per la ricerca vettoriale: {eval_results_vector['mrr']:.4f}")
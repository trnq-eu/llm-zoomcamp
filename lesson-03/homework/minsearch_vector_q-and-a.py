from minsearch import VectorSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import requests
from minsearch_text_2 import hit_rate, mrr, evaluate
import pandas as pd


# Define the URL prefix for the data
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'

# Fetch documents with IDs
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

# Fetch ground truth data
ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
ground_truth = df_ground_truth.to_dict(orient='records')

texts = []

for doc in documents:
    t = doc['question'] + ' ' + doc['text']
    texts.append(t)

pipeline = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)
X = pipeline.fit_transform(texts)

# vector search for question

vindex = VectorSearch(keyword_fields={'course'})
vindex.fit(X, documents)



def vector_search_function(query, course):
    query_vector = pipeline.transform([query])[0] # to vectorize the query uses the same pipeline we used for the docs
    filter_dict = {'course': course}

    results = vindex.search(
        query_vector,
        filter_dict=filter_dict,
        num_results=5
    )
    return results

# --- Valuta la funzione di ricerca vettoriale ---
print("Valutazione della ricerca vettoriale:")
eval_results_vector = evaluate(ground_truth, vector_search_function)

# Stampa i risultati
print(f"Hit rate per la ricerca vettoriale: {eval_results_vector['hit_rate']:.4f}") # 0.82
print(f"MRR per la ricerca vettoriale: {eval_results_vector['mrr']:.4f}")
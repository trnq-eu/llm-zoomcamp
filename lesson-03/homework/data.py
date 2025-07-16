import requests
import pandas as pd
from tqdm.auto import tqdm
from minsearch import Index


url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
ground_truth = df_ground_truth.to_dict(orient='records')

# Create and fit the index
index = Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
index.fit(ground_truth)

# code for evaluating retrieval


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

boost = {'question': 1.5, 'section': 0.1}

def minsearch_search(q):
    """
    Performs a search using minsearch with specified boosting and filtering.
    """
    query = q['question']
    results = index.search(
        query=query,
        boost_dict=boost, # Apply the defined boosting parameters
        filter_dict={'course': 'machine-learning-zoomcamp'} # Filter by course
    )
    return results

# Perform the evaluation
eval_results = evaluate(ground_truth, minsearch_search)

# Print the hit rate as requested
print(f"Hit rate for minsearch with boosting: {eval_results['hit_rate']:.4f}")

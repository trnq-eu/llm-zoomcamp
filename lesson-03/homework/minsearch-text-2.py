import requests
import pandas as pd
from tqdm.auto import tqdm
import minsearch
from minsearch import Index


# Define the URL prefix for the data
url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'

# Fetch documents with IDs
docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
documents = requests.get(docs_url).json()

# Fetch ground truth data
ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
df_ground_truth = pd.read_csv(ground_truth_url)
ground_truth = df_ground_truth.to_dict(orient='records')



# --- Functions for evaluating retrieval ---

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

# Create and fit the minsearch index
index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course", "id"] # Now 'id' is explicitly included in returned fields
)
index.fit(documents) # Fit with 'documents'


def minsearch_search(query, course):
    """
    Performs a search using minsearch with specified boosting and filtering.
    """
    boost = {'question': 1.5, 'section': 0.1}
    query = query
    results = index.search(
        query=query,
        boost_dict=boost, # Apply the defined boosting parameters
        filter_dict={'course':course}, # Filter by course
        num_results=5
    )
    return results

eval_results = evaluate(ground_truth, minsearch_search)

# Print the hit rate as requested
print(f"Hit rate for minsearch with boosting: {eval_results['hit_rate']:.4f}")

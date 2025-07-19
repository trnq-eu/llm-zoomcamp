import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np


url = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/refs/heads/main/03-evaluation/rag_evaluation/data/results-gpt4o-mini.csv'

response = requests.get(url)

save_dir = 'cached_data'

# The file download and saving part is commented out, assuming the file is already in 'cached_data'
# if response.status_code == 200:
#     with open(f"{save_dir}/results-gpt40-mini.csv", "wb") as f:
#         f.write(response.content)
#         print("File salvato come results-gpt4o-mini.csv")
# else:
#     print(f"Errore nel download: {response.status_code}")

df_results = pd.read_csv('cached_data/results-gpt40-mini.csv')

pipeline = make_pipeline(
    TfidfVectorizer(min_df=3),
    TruncatedSVD(n_components=128, random_state=1)
)

# Fit the pipeline to all the texts
all_text = df_results.answer_llm + ' ' + df_results.answer_orig + ' ' + df_results.question
pipeline.fit(all_text)

# Transform single csv columns to get embeddings
embeddings_answer_llm = pipeline.transform(df_results.answer_llm.fillna(''))
embeddings_question = pipeline.transform(df_results.question.fillna(''))
embeddings_answer_orig = pipeline.transform(df_results.answer_orig.fillna(''))

def cosine(u, v):
    u_norm = np.sqrt(u.dot(u))
    v_norm = np.sqrt(v.dot(v))
    # Handle division by zero for zero vectors
    if u_norm == 0 or v_norm == 0:
        return 0.0 # Or np.nan, depending on how you want to handle zero vectors
    return u.dot(v) / (u_norm * v_norm)

# calculate cosine similarity
cosine_similarities_llm_q = []
cosine_similarities_llm_orig = []
cosine_similarities_q_orig = []

for i in range(len(df_results)):
    # similarity between llm answer and question
    sim_llm_q = cosine(embeddings_answer_llm[i], embeddings_question[i])
    cosine_similarities_llm_q.append(sim_llm_q)

    # similarity between llm answer and original answer
    sim_llm_orig = cosine(embeddings_answer_llm[i], embeddings_answer_orig[i])
    cosine_similarities_llm_orig.append(sim_llm_orig) # <-- CORRECTED THIS LINE

    # similarity between question and original answer
    sim_q_orig = cosine(embeddings_question[i], embeddings_answer_orig[i])
    cosine_similarities_q_orig.append(sim_q_orig)

# Add the cosine similarity scores as new columns to the DataFrame
df_results['cosine_sim_llm_q'] = cosine_similarities_llm_q
df_results['cosine_sim_llm_orig'] = cosine_similarities_llm_orig
df_results['cosine_sim_q_orig'] = cosine_similarities_q_orig

print("\nDataFrame with Cosine Similarities:")
print(df_results[['answer_llm', 'question', 'answer_orig', 'cosine_sim_llm_q', 'cosine_sim_llm_orig', 'cosine_sim_q_orig']].head())

# Compute the average similarities
print(f"\nAverage Cosine Similarity (LLM Answer vs Question): {df_results['cosine_sim_llm_q'].mean():.4f}")
print(f"Average Cosine Similarity (LLM Answer vs Original Answer): {df_results['cosine_sim_llm_orig'].mean():.4f}")
print(f"Average Cosine Similarity (Question vs Original Answer): {df_results['cosine_sim_q_orig'].mean():.4f}")
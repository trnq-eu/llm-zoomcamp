import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from rouge import Rouge

rouge_scorer = Rouge()


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

r = df_results
rouge_1_f1_scores = [] 


for i, row in df_results.iterrows():
    llm_answer = row['answer_llm']
    reference = row['answer_orig']

    scores = rouge_scorer.get_scores(llm_answer, reference)
    rouge_scores = scores[0]

    rouge_1_f1 = rouge_scores['rouge-1']['f']
    rouge_1_f1_scores.append(rouge_1_f1)

df_results['rouge_1_f1'] = rouge_1_f1_scores

average_rouge_1_f1 = df_results['rouge_1_f1'].mean(skipna=True)


print(f"\nMedia del punteggio ROUGE-1 F1 (LLM Answer vs Original Answer): {average_rouge_1_f1:.4f}")


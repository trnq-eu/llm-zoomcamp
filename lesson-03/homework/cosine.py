import requests

url = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/refs/heads/main/03-evaluation/rag_evaluation/data/results-gpt4o-mini.csv'

response = requests.get(url)

save_dir = 'cached_data'

if response.status_code == 200:
    with open(f"{save_dir}/results-gpt40-mini.csv", "wb") as f:
        f.write(response.content)
        print("File salvato come results-gpt4o-mini.csv")
else:
    print(f"Errore nel download: {response.status_code}")


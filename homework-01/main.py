import requests
from elasticsearch import Elasticsearch
from tqdm import tqdm


# retrieve documents

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# index documents

index_settings ={
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}

index_name = "course-faq"

# create elasticsearch client
es_client = Elasticsearch('http://localhost:9200')

# # create index
# es_client.indices.create(index=index_name, body=index_settings)

# # add documents into the es index
# for doc in tqdm(documents):
#      es_client.index(index=index_name, document=doc)

q = "How do copy a file to a Docker container?"

def elastic_search(query):

    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)

    result_docs = []

    for hit in response['hits']['hits']:
        result_docs.append({
            '_score': hit['_score'],
            'document' : hit['_source']
            })

    return result_docs

# print(elastic_search(q)[0:3])

def build_prompt(query, search_results):
    context_template = """
    Q: {question}
    A: {text}
    """.strip()
import requests
from elasticsearch import Elasticsearch
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import tiktoken


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

q = "How do I execute a command in a running docker container?"

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


def build_prompt(query, search_results):
    context_template = """
    Q: {question}
    A: {text}
    """.strip()

    context_entries = []
    for result in search_results:
        context_entry = context_template.format(question=result['document']['question'], text=result['document']['text'])
        context_entries.append(context_entry)

    context = "\n\n".join(context_entries)
    return context

prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()


search_result = elastic_search(q)

llm_context = build_prompt(q, search_result)
final_prompt = prompt_template.format(question=q, context=llm_context)

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

load_dotenv(dotenv_path)

openai_api_key = os.getenv('OPENAI')

client = OpenAI(api_key=openai_api_key)

def llm(prompt):
    response = client.chat.completions.create(
        model = 'gpt-4o',
        messages = [{"role": "user",
                     "content": prompt}],

    )

    return response.choices[0].message.content

encoding = tiktoken.encoding_for_model("gpt-4o")

prompt_tokens = encoding.encode(final_prompt)
num_prompt_tokens = len(prompt_tokens)
print(f"The prompt has {num_prompt_tokens} tokens.")
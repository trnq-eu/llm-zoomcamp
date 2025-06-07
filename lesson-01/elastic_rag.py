from elasticsearch import Elasticsearch
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

load_dotenv(dotenv_path)

openai_api_key = os.getenv('OPENAI')

client = OpenAI(api_key=openai_api_key)

q = "The course is already started, can I still enroll?"


# create elasticsearch client
es_client = Elasticsearch('http://localhost:9200')
print(es_client.info())

#

with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)

documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

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

index_name = "course-questions"
# ObjectApiResponse({'acknowledged': True, 'shards_acknowledged': True, 'index': 'course-questions'})


# create index
# es_client.indices.create(index=index_name, body=index_settings)

# add documents into the es index
# for doc in tqdm(documents):
#     es_client.index(index=index_name, document=doc)

query = "how do I run kafka?"

def elastic_search(query):

    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)

    result_docs = []

    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs

# # results = search('how do i run kafka?')


def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant. 
    Answer the QUESTION based on the CONTEXT.
    If the CONTEXT doesn't contain the answer, output NONE.

    QUESTION: {question}
    CONTEXT: {context}

    """.strip()

    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt


def llm(prompt):
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages = [{"role": "user",
                     "content": prompt}],

    )

    return response.choices[0].message.content



query = 'how do I run kafka?'

def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

print(rag("how do i run kafka?"))


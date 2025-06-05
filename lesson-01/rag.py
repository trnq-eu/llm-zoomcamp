import minsearch
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

load_dotenv(dotenv_path)

openai_api_key = os.getenv('OPENAI')

client = OpenAI(api_key=openai_api_key)

q = "The course is already started, can I still enroll?"




# response = client.chat.completions.create(
#     model = 'gpt-4o-mini',
#     messages = [{"role": "user",
#                  "content": q}],
    
# )



# print(response)


with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)

documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

index = minsearch.Index(
    text_fields=['question', 'text', 'section'],
    keyword_fields=['course']
)

# print(documents[0])


index.fit(documents)

boost = {'question': 3.0, 
         'section': 0.5}

results = index.search(
    query = q,
    filter_dict = {'course':'data-engineering-zoomcamp'},
    boost_dict = boost,
    num_results = 5

)

prompt_template = """
You're a course teaching assistant. 
Answer the QUESTION based on the CONTEXT.
If the CONTEXT doesn't contain the answer, output NONE.

QUESTION: {question}
CONTEXT: {context}

""".strip()

context = ""

for doc in results:
    context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"


prompt = prompt_template.format(question=q, context=context).strip()

response = client.chat.completions.create(
    model = 'gpt-4o-mini',
    messages = [{"role": "user",
                 "content": prompt}],
    
)

print(response.choices[0].message.content)
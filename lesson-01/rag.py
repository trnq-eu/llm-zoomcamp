import minsearch
import json


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

q = "The course is already started, can I still enroll?"

index.fit(documents)

boost = {'question': 3.0, 
         'section': 0.5}

results = index.search(
    query = q,
    filter_dict = {'course':'data-engineering-zoomcamp'},
    boost_dict = boost,
    num_results = 5

)

print(results)
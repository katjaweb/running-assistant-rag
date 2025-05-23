"""
Elasticsearch Indexing Script for Running Assistant FAQ Documents

This script initializes and populates an Elasticsearch index (`running-questions`) with structured 
FAQ documents related to running.

Functionality includes:
- Loading documents from a JSON file (`documents.json`)
- Defining index settings and mappings tailored for text-based search.
- Indexing all documents into Elasticsearch using the `tqdm` progress bar for visual feedback.

This indexing process prepares the data for use in a Retrieval-Augmented Generation (RAG) 
pipeline for a personal running assistant.
"""

import json
from tqdm import tqdm
from elasticsearch import Elasticsearch

INDEX_NAME = "running-questions"

es_client = Elasticsearch('http://localhost:9200')

with open('documents.json', 'rt', encoding="utf-8") as f_in:
    docs_raw = json.load(f_in)

documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

index_settings = {
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

es_client.indices.create(index=INDEX_NAME, body=index_settings)

for doc in tqdm(documents):
    es_client.index(index=INDEX_NAME, document=doc)

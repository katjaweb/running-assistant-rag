"""
Module to initialize and populate a Qdrant vector database collection for a running FAQ dataset.

This script performs the following steps:
1. Connects to a local Qdrant instance.
2. Loads a JSON file (`documents.json`) containing course-based question-answer documents.
3. Extracts and flattens the documents into a unified list, adding course metadata.
4. Deletes any existing Qdrant collection named 'running-faq'.
5. Creates a new Qdrant collection with vector configuration based on cosine similarity
    and a specified embedding dimensionality.
6. Encodes the text data using a specified embedding model
    (handle: 'jinaai/jina-embeddings-v2-small-en').
7. Inserts the resulting vector representations into the Qdrant collection along with
    associated payloads.

Constants:
- `EMBEDDING_DIMENSIONALITY`: Dimensionality of the vector embeddings (512).
- `MODEL_HANDLE`: Identifier of the model used for embedding the documents.
- `COLLECTION_NAME`: Name of the Qdrant collection to be created and populated.

Requirements:
- Qdrant must be running locally on port 6333.
- The file `documents.json` must be structured with a list of question-answer documents.

Note:
This script clears and recreates the collection each time it is run. Any existing data
in the specified collection will be lost.
"""

import json
from qdrant_client import QdrantClient, models

qd_client = QdrantClient("http://localhost:6333")

EMBEDDING_DIMENSIONALITY = 512
MODEL_HANDLE = "jinaai/jina-embeddings-v2-small-en"
COLLECTION_NAME = "running-faq"

# load the running qa documents
with open('documents.json', 'rt', encoding='utf-8') as f_in:
    docs_raw = json.load(f_in)

    documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

qd_client.delete_collection(collection_name=COLLECTION_NAME)

qd_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIMENSIONALITY,
        distance=models.Distance.COSINE
    )
)

points = []

for i, doc in enumerate(documents):
    text = doc['question'] + ' ' + doc['text']
    vector = models.Document(text=text, model=MODEL_HANDLE)
    point = models.PointStruct(
        id=i,
        vector=vector,
        payload=doc
    )
    points.append(point)

qd_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

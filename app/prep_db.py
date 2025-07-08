"""
Qdrant Indexing Script

This module handles the initialization and population of a Qdrant vector database
collection with embedded documents for semantic search. It loads question-answer
documents and ground truth data, generates vector embeddings using a specified model,
creates or resets the Qdrant collection, and upserts the document embeddings with
associated metadata. Additionally, it initializes the PostgreSQL database schema.

Key functionalities:
- Load documents and ground truth data
- Initialize and configure Qdrant collection
- Build and upsert vector points into Qdrant
- Logging of the indexing process steps
- Database initialization

Requires environment variables for configuration
(e.g., QDRANT_URL_LOCAL, MODEL_NAME, COLLECTION_NAME).
"""

import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv

from db import init_db
from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "../data"))

QDRANT_URL = os.getenv("QDRANT_URL_LOCAL")
MODEL_HANDLE = os.getenv("MODEL_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_DIMENSIONALITY = 512


def load_documents():
    """
    Load question-answer documents from a JSON file.

    Returns:
        list: A list of documents loaded from 'docs-with-ids.json', 
              each containing question, text, section and id fields.
    """
    with open(f'{DATA_PATH}/docs-with-ids.json', 'rt', encoding='utf-8') as f_in:
        documents = json.load(f_in)
    return documents


def load_ground_truth():
    """
    Load ground truth data from a CSV file into a list of dictionaries.

    Returns:
        list: A list of records representing ground truth data loaded from 'ground-truth-data.csv'.
    """
    df_ground_truth = pd.read_csv(f'{DATA_PATH}/ground-truth-data.csv')
    ground_truth = df_ground_truth.to_dict(orient="records")
    return ground_truth


def initialize_qdrant_collection(
        collection_name=COLLECTION_NAME,
        url = QDRANT_URL,
        embedding_dim: int = EMBEDDING_DIMENSIONALITY
        ):
    """
    Initialize a Qdrant collection by deleting any existing one and creating a new collection 
    configured for vector search with the specified embedding dimensionality.

    Args:
        collection_name (str): Name of the Qdrant collection to initialize.
        url (str): URL of the Qdrant server.
        embedding_dim (int): Dimensionality of the vector embeddings.

    Returns:
        QdrantClient: An instance of the initialized Qdrant client connected to the collection.
    """
    qd_client = QdrantClient(url)

    qd_client.delete_collection(collection_name=collection_name)

    qd_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE
        )
    )

    logging.info("Qdrant collection %s has been created and initialized.", collection_name)

    return qd_client


def build_qdrant_points(documents, model_handle):
    """
    Generate Qdrant Point objects by embedding combined question and text fields from documents.

    Args:
        documents (list): List of document dicts, each containing 'question' and 'text' fields.
        model_handle (str): Model identifier used to generate embeddings.

    Returns:
        list: A list of Qdrant Point instances ready for upserting into the collection.
    """
    points = []

    for i, doc in enumerate(documents):
        text = doc['question'] + ' ' + doc['text']
        vector = models.Document(text=text, model=model_handle)

        point = models.PointStruct(
            id=i,
            vector=vector,
            payload=doc
        )

        points.append(point)

    return points


def upsert_documents_to_qdrant(points, qd_client, collection_name):
    """
    Upsert a list of document points into the specified Qdrant collection.

    Args:
        points (list): List of Qdrant PointStruct objects to be inserted or updated.
        qd_client (QdrantClient): The Qdrant client instance used to interact with the database.
        collection_name (str): Name of the target Qdrant collection.
    """
    qd_client.upsert(
        collection_name=collection_name,
        points=points
    )
    logging.info("Upserted %s documents to Qdrant collection %s.", len(points), collection_name)


def main():
    """
    Main execution function that orchestrates the entire Qdrant indexing process.

    It loads documents and ground truth data, initializes the Qdrant collection,
    builds and upserts document embeddings, and initializes the database schema.
    Logs progress and completion status throughout the process.
    """
    logging.info("Starting Qdrant indexing process...")

    documents = load_documents()
    logging.info("Loaded %s documents.", len(documents))

    ground_truth = load_ground_truth()
    logging.info("Loaded %s ground truth.", len(ground_truth))

    qd_client = initialize_qdrant_collection(collection_name=COLLECTION_NAME)

    points = build_qdrant_points(documents, model_handle=MODEL_HANDLE)
    logging.info("Built %s Qdrant points.", len(points))

    upsert_documents_to_qdrant(points, qd_client, collection_name=COLLECTION_NAME)
    logging.info("Qdrant indexing process completed successfully.")

    logging.info("Initializing database...")
    init_db()

    logging.info("Indexing process completed successfully!")


if __name__ == "__main__":
    main()

# pgcli Connection Command for Local Database
# pgcli -h localhost -U your_username -d running_assistant -W

# list all databases
# \l

# connect to running_assistant database
# \c running_assistant

# list all tables in the current database
# \dt

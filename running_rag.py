"""
Running Assistant – A Retrieval-Augmented Generation (RAG) App for Running-Related Questions

This module implements a Streamlit-based personal running assistant that answers user questions 
by combining document retrieval with generative AI. It includes the following components:

- `elastic_search`: Searches an Elasticsearch index for relevant FAQ entries related to running.
- `build_prompt`: Constructs a prompt using the retrieved documents and the user's question.
- `llm`: Sends the prompt to the GPT-4o language model and retrieves the generated response.
- `rag_pipeline`: Orchestrates the full RAG workflow — from retrieval to answer generation.
- `main`: Launches the Streamlit app interface for interacting with the assistant.

Intended for use as an interactive FAQ or personal coach for runners.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from elasticsearch import Elasticsearch
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

es_client = Elasticsearch('http://localhost:9200')


def elastic_search(query, index_name = "running-questions"):
    """
    Performs a search in the specified Elasticsearch index and returns relevant documents.

    The search uses a weighted multi-field match on the fields 'question', 'text', and 'section', 
    with an additional filter to include only documents related to the 'running-assistant-rag'
    course.

    Args:
        query (str): The search query string.
        index_name (str, optional): The name of the Elasticsearch index. Defaults to
        'running-questions'.

    Returns:
        list[dict]: A list of documents (as dictionaries) that match the search criteria.
    """
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
                        "course": "running-assistant-rag"
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


def build_prompt(query, search_results):
    """
    Constructs a prompt for a language model based on a user question and search results.

    Formats a structured prompt using a predefined template that includes the question and 
    a context section composed of relevant FAQ entries. Each entry includes section, question, 
    and answer details to guide the model in generating a comprehensive response.

    Args:
        query (str): The user's question.
        search_results (list[dict]): A list of search result documents containing 'section', 
                                     'question', and 'text' fields.

    Returns:
        str: A formatted prompt string ready for use with a language model.
    """

    prompt_template = """
You are a personal trainer specialized in running.

Answer the QUESTION at the end based on the full CONTEXT below, which contains multiple relevant FAQ entries.
Use **all available context entries** to form your answer. If multiple answers apply, summarize or combine them clearly.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context_entries = []

    for i, doc in enumerate(search_results, start=1):
        entry = (
            f"[Entry {i}]\n"
            f"Section: {doc['section']}\n"
            f"Question: {doc['question']}\n"
            f"Answer: {doc['text']}"
        )
        context_entries.append(entry)

    context = "\n\n".join(context_entries)

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    """
    Sends a prompt to the GPT-4o language model and returns the generated response.

    Uses specified generation parameters such as temperature and top_p to control creativity 
    and diversity of the output.

    Args:
        prompt (str): The input prompt to be processed by the language model.

    Returns:
        str: The generated response from the language model.
    """
    # print(prompt)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}],
        temperature=1.5,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def rag_pipeline(query):
    """
    Executes a full Retrieval-Augmented Generation (RAG) pipeline to answer a user query.

    Retrieves relevant documents using Elasticsearch, constructs a prompt with the retrieved
    context, and generates a final answer using a language model.

    Args:
        query (str): The user's input question.

    Returns:
        str: The generated answer based on retrieved context and the language model's response.
    """
    context_docs = elastic_search(query)
    prompt = build_prompt(query, context_docs)
    answer = llm(prompt)
    return answer


def set_custom_styles():
    """
    custom style for streamlit UI
    """
    st.markdown(
        """
        <style>
        /* Button color */
        div.stButton > button {
            background-color: #4CAF50; /* grün */
            color: white;
            border: none;
            padding: 0.6em 1.2em;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 1em;
            margin: 0.4em 0;
            cursor: pointer;
            border-radius: 8px;
            transition-duration: 0.3s;
        }

        div.stButton > button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    """
    Launches the Streamlit app for the personal running assistant.

    Sets up the UI, captures user input, and triggers the RAG pipeline to generate and display 
    an answer to the user's question.
    """
    set_custom_styles()

    st.title("Dies ist Ihr persönlicher Laufassistent")

    user_input = st.text_input("Geben Sie hier Ihre Frage ein:")

    if st.button("Frage stellen"):
        with st.spinner('🏃‍♂️ Läuft los…'):
            output = rag_pipeline(user_input)
            st.success("Antwort")
            st.write(output)

if __name__ == "__main__":
    main()

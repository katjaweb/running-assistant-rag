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
import time
import json
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models
from openai import OpenAI

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
MODEL_HANDLE = os.getenv('MODEL_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
QDRANT_URL = os.getenv('QDRANT_URL', 'http://qdrant:6333')

qd_client = QdrantClient(QDRANT_URL)
openai_client = OpenAI(api_key=api_key)


def vector_search(question):
    """
    Performs a vector similarity search in the Qdrant collection using the given question.

    Embeds the input question using the specified model and retrieves the top 5 most similar
    documents from the 'running-faq' collection based on cosine similarity. Returns a list
    of payloads (metadata and content) from the matched documents.
    """
    print('vector_search is used')

    query_points = qd_client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(
            text=question,
            model=MODEL_HANDLE
        ),
        limit=5,
        with_payload=True
    )

    results = []

    for point in query_points.points:
        results.append(point.payload)

    print(results)

    return results


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

def llm(prompt, model):
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
    start_time = time.time()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.5,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content
    tokens = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }

    end_time = time.time()
    response_time = end_time - start_time

    return answer, tokens, response_time


def evaluate_relevance(question, answer):
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt, model='gpt-4o')

    try:
        json_eval = json.loads(evaluation)
        return json_eval['Relevance'], json_eval['Explanation'], tokens
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", tokens


def calculate_openai_cost(tokens):
    openai_cost = 0

    openai_cost = (tokens['prompt_tokens'] * 0.03 + tokens['completion_tokens'] * 0.06) / 1000

    return openai_cost


def rag_pipeline(query, model='gpt-4o'):
    """
    Executes a full Retrieval-Augmented Generation (RAG) pipeline to answer a user query.

    Retrieves relevant documents using Elasticsearch, constructs a prompt with the retrieved
    context, and generates a final answer using a language model.

    Args:
        query (str): The user's input question.

    Returns:
        str: The generated answer based on retrieved context and the language model's response.
    """
    context_docs = vector_search(query)
    prompt = build_prompt(query, context_docs)
    answer, tokens, response_time = llm(prompt, model)

    relevance, explanation, eval_tokens = evaluate_relevance(query, answer)

    openai_cost = calculate_openai_cost(tokens)

    return {
        'answer': answer,
        'response_time': response_time,
        'relevance': relevance,
        'relevance_explanation': explanation,
        'prompt_tokens': tokens['prompt_tokens'],
        'completion_tokens': tokens['completion_tokens'],
        'total_tokens': tokens['total_tokens'],
        'eval_prompt_tokens': eval_tokens['prompt_tokens'],
        'eval_completion_tokens': eval_tokens['completion_tokens'],
        'eval_total_tokens': eval_tokens['total_tokens'],
        'openai_cost': openai_cost
    }

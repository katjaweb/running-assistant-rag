# Running Assistant – A Retrieval-Augmented Generation (RAG) App for Running-Related Questions

![running](./images/running-6660187_640.jpg)

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to assist users with questions related to running and training. It provides information on running techniques, training plans, recovery strategies, and more. The application is intended for demonstration and experimentation purposes and serves as an interactive FAQ or personal coach for runners.

The Running Assistant App for Running-Related Questions implements a Streamlit-based personal running assistant that answers user questions by combining document retrieval with generative AI. It includes the following components:

- Retrieves relevant FAQ entries from an Elasticsearch index based on the user's query.
- Builds a prompt by combining the retrieved documents with the user's question.
- Sends the prompt to the GPT-4o language model to generate a coherent response.
- Orchestrates the full RAG pipeline — from retrieval to answer generation.
- Provides an interactive interface via a Streamlit web app for interacting with the assistant.

The underlying FAQ document used for retrieval is written in German and were created using the GPT-4o model.

# Get started

Follow these steps to set up and run the Running Assistant App:

**Install dependencies**

Use `make setup` to install `pipenv` and all required Python packages:

```bash
make setup
```

**Launch a local Elasticsearch container using Docker**

```bash
make run_elasticsearch
```

Wait a few seconds for Elasticsearch to fully initialize before continuing.
You can test if it’s running with:

```bash
curl http://localhost:9200
```

**Activate Python Environment**

Enter the virtual environment created by `pipenv`:

```bash
pipenv shell
```

**Set Your OpenAI API Key**

To use the RAG application, you need an OpenAI API key. You can create one in your OpenAI account dashboard.

Once created, export the key as an environment variable:

```bash
export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
```

**Index Running FAQ Documents**

Populate the `running-questions` index in Elasticsearch with structured documents:

```bash
make es_indexing
```

**Launch the Running Assistant App**

Start the Streamlit-based interface for your personal running assistant:

```bash
make running_assistant
```

This will automatically open a new page in your web browser.
There, you can enter your questions and receive context-based answers related to running.

![rag-app](./images/rag-app.png)

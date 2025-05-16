# running-assistant-rag
A Retrieval-Augmented Generation (RAG) chatbot focused on running and training. This project aims to answer questions about running techniques, training plans, recovery, and more. Built for demonstration and experimentation.

Download Anaconda for Linux
in workspaces installienern, nicht in workspaces/running-assistant-rag
cd..
'''bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
'''

Install Ananconda:
cd /workspaces
'''bash
bash Anaconda3-2024.10-1-Linux-x86_64.sh
'''

with anaconda, you don't need to install jupyter, pandas, tqdm, scikit-learn

install elasticsearch and openai
'''bash
pip install elasticsearch openai
'''

Installed numpy 1.26.4 instead of 2.2.4 due to an error while importing elaticsearch

Setup ollama by running this command:
'''bash
curl -fsSL https://ollama.com/install.sh | sh
'''

# Ollama

for more information how to download for windows and MacOS visit github repo: https://github.com/ollama/ollama

For starting the server
'''bash
ollama start
'''

To run and chat with phi4-mini
'''bash
ollama run phi4-mini
'''

'''bash
ollama run phi4-mini
'''

Connecting to OpenAI API:
'''bash
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)
'''

# Elasticsearch

Running ElasticSearch:
'''bash
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:9.0.1
'''

Do make sure that the needed ports 9200 and 9300 are running
'''bash
curl http://localhost:9200

Index settings:
'''bash
{
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
'''

To run Elasticsearch and ollama in docker:
'''bash
docker-compose up
'''

Load the model
'''bash
docker exec -it ollama bash
'''

and then
'''bash
ollama pull phi4-mini
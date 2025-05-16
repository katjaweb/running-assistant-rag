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

Setup ollama by running this command:
'''bash
curl -fsSL https://ollama.com/install.sh | sh
'''

for more information how to download for windows and MacOS visit github repo: https://github.com/ollama/ollama

For starting do
'''bash
ollama start
'''

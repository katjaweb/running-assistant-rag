run_elasticsearch:
	docker run -it -d \
	--rm \
	--name elasticsearch \
	-m 4GB \
	-p 9200:9200 \
	-p 9300:9300 \
	-v elasticsearch_data:/usr/share/elasticsearch/data \
	-e "discovery.type=single-node" \
	-e "xpack.security.enabled=false" \
	docker.elastic.co/elasticsearch/elasticsearch:9.0.1

es_indexing:
	pipenv run python es_indexing.py

running_assistant:
	pipenv run streamlit run running_rag.py

setup:
	pip install pipenv
	pipenv install --dev

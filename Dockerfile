FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

COPY data/docs-with-ids.json ./data/docs-with-ids.json
COPY data/ground-truth-data.csv ./data/ground-truth-data.csv

CMD ["streamlit", "run", "app.py"]

# docker build -f app/Dockerfile -t streamlit-app .

# docker run -p 8501:8501 streamlit-app
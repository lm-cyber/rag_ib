FROM python:3.12-slim

WORKDIR /opt/app-root/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --prefer-binary nltk spacy

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN python -m spacy download ru_core_news_sm

RUN python -m nltk.downloader stopwords punkt punkt_tab

COPY requirements.txt /opt/app-root/requirements.txt
RUN pip install --no-cache-dir -r /opt/app-root/requirements.txt

ENV PYTHONPATH /opt/app-root/src/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7420"]

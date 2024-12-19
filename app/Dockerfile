FROM python:3.12-slim

WORKDIR /opt/app-root/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --prefer-binary nltk spacy

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN python -m spacy download ru_core_news_sm

RUN python -m nltk.downloader stopwords punkt punkt_tab

COPY requirements.txt /opt/app-root/requirements.txt
RUN pip install --no-cache-dir -r /opt/app-root/requirements.txt

ENV PYTHONPATH /opt/app-root/src/app
ENV PYTHONUNBUFFERED=1

COPY entrypoint.sh /opt/app-root
RUN chmod +x /opt/app-root/entrypoint.sh

COPY . /opt/app-root/src

CMD ["/opt/app-root/entrypoint.sh"]
import os
from datetime import date
from enum import Enum
from time import perf_counter
from typing import List, Optional

import bm25s
import chromadb
import feedparser
import polars as pl
from chromadb import EmbeddingFunction, Documents, Embeddings
from fastapi import APIRouter
from fastapi import HTTPException
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain_voyageai import VoyageAIRerank
from pydantic import BaseModel

from api.config.model_config import model_config
from api.preprocess import preprocess_text

base_dir = "/opt/app-root/src/app/data/news"


class VoyageEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        self.embeddings = VoyageAIEmbeddings(
            voyage_api_key=model_config.voyage_api_key,
            model=model_config.voyage_embeddings_model_name
        )

    def __call__(self, input: Documents) -> Embeddings:
        return self.embeddings.embed_documents(texts=input)

    def embed_query(self, query: str) -> Embeddings:
        return self.embeddings.embed_documents([query])


voyage_embeddings_function = VoyageEmbeddingFunction()
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
collection = chroma_client.get_or_create_collection("news", embedding_function=voyage_embeddings_function, metadata={"hnsw:space": "cosine"})


class SOURCE(str, Enum):
    LENTA = "lenta"
    RBC = "rbc"
    TASS = "tass"
    RIA = "ria"
    VEDOMOSTI = "vedomosti"
    ALL = "all"


source_mapping = {
    SOURCE.LENTA: "https://lenta.ru/rss/news",
    SOURCE.RBC: "https://rssexport.rbc.ru/rbcnews/news/20/full.rss",
    SOURCE.TASS: "https://tass.ru/rss/v2.xml",
    SOURCE.RIA: "https://ria.ru/export/rss2/archive/index.xml",
    SOURCE.VEDOMOSTI: "https://www.vedomosti.ru/rss/news",
}


def parse_rss(url, source):
    feed = feedparser.parse(url)
    parsed_entries = []
    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", "") or entry.get("description", "") or entry.get("value", "")
        if summary:
            title = title + " " + summary
        parsed_entries.append({
            "text": title,
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "source": source,
        })
    return parsed_entries


def parse_all_sources():
    all_entries = []
    for s, url in source_mapping.items():
        if s != SOURCE.ALL:
            all_entries.extend(parse_rss(url, s))
    return all_entries


def embed_documents_to_collection(collection, documents, embedding_function):
    start = perf_counter()

    ids = [doc["metadata"]["id"] for doc in documents]

    if ids:
        existing_documents = collection.get(ids=ids, include=["metadatas"])["metadatas"]
    else:
        existing_documents = []

    existing_ids = {metadata['id'] for metadata in existing_documents if metadata is not None}

    new_documents = [doc for doc in documents if doc["metadata"]["id"] not in existing_ids]

    if not new_documents:
        return

    ids_new = [doc["metadata"]["id"] for doc in new_documents]
    documents_new = [doc["page_content"] for doc in new_documents]
    metadatas_new = [doc["metadata"] for doc in new_documents]

    embeddings = embedding_function(documents_new)

    collection.add(ids=ids_new, documents=documents_new, metadatas=metadatas_new, embeddings=embeddings)
    print(f'{collection.name} added {len(ids_new)} new documents in {perf_counter() - start:.2f}s')


router = APIRouter()


@router.get("/rss/{source}", response_model=List[dict])
async def get_rss(source: SOURCE):
    print('parsing')
    if source == SOURCE.ALL:
        parsed_entries = parse_all_sources()
    else:
        url = source_mapping[source]
        parsed_entries = parse_rss(url, source)

    if not parsed_entries:
        return []

    df = (
        pl.DataFrame(parsed_entries)
        .with_columns(
            pl.col("published").str.strptime(pl.Datetime, "%a, %d %b %Y %H:%M:%S %z", strict=False).alias("published")
        )
        .with_columns(
            pl.col("published").cast(pl.Date).alias("date")
        )
    )

    unique_dates = df["date"].unique().to_list()
    os.makedirs(base_dir, exist_ok=True)

    needed_cols = ["text", "link", "published", "source", "date"]

    all_docs_for_chroma = []

    for d in unique_dates:
        df_current = df.filter(pl.col("date") == d)
        file_path = os.path.join(base_dir, f"{d}.parquet")

        for col in needed_cols:
            if col not in df_current.columns:
                df_current = df_current.with_columns(pl.lit(None).alias(col))

        if os.path.exists(file_path):
            df_existing = pl.read_parquet(file_path)

            for col in needed_cols:
                if col not in df_existing.columns:
                    df_existing = df_existing.with_columns(pl.lit(None).alias(col))

            if "text_cleaned" not in df_existing.columns and "text_cleaned" not in df_current.columns:
                df_existing = df_existing.with_columns(pl.lit(None).alias("text_cleaned"))
                df_current = df_current.with_columns(pl.lit(None).alias("text_cleaned"))
            elif "text_cleaned" in df_existing.columns and "text_cleaned" not in df_current.columns:
                df_current = df_current.with_columns(pl.lit(None).alias("text_cleaned"))
            elif "text_cleaned" not in df_existing.columns and "text_cleaned" in df_current.columns:
                df_existing = df_existing.with_columns(pl.lit(None).alias("text_cleaned"))

            df_combined = pl.concat([df_existing, df_current]).unique(subset=["link"])

            mask_no_clean = df_combined["text_cleaned"].is_null()

            df_combined = df_combined.with_columns(
                pl.when(mask_no_clean)
                .then(pl.col("text").map_elements(preprocess_text, return_dtype=pl.Utf8))
                .otherwise(pl.col("text_cleaned"))
                .alias("text_cleaned")
            )
        else:
            df_combined = df_current.with_columns(
                pl.col("text").map_elements(preprocess_text, return_dtype=pl.Utf8).alias("text_cleaned")
            )

        df_combined.write_parquet(file_path)

        df_dicts = df_combined.to_dicts()
        docs = []
        for row in df_dicts:
            docs.append({
                "page_content": row.get("text_cleaned", "") or "",
                "metadata": {
                    "id": row.get("link", ""),
                    "date": str(row.get("date", "")),
                    "source": row.get("source", ""),
                    "text": row.get("text", ""),
                }
            })
        all_docs_for_chroma.extend(docs)

    if all_docs_for_chroma:
        embed_documents_to_collection(
            collection=collection,
            documents=all_docs_for_chroma,
            embedding_function=voyage_embeddings_function
        )

    return parsed_entries


@router.get("/available_dates", response_model=List[date])
async def get_available_dates():
    if not os.path.exists(base_dir):
        raise HTTPException(status_code=404, detail="Data directory does not exist.")

    files = os.listdir(base_dir)
    dates_found = []

    for f in files:
        if f.endswith(".parquet"):
            date_str = f[:-8]  # remove ".parquet"
            try:
                parsed_date = date.fromisoformat(date_str)
                dates_found.append(parsed_date)
            except ValueError:
                pass

    if not dates_found:
        return []

    return sorted(dates_found)


class SearchRequest(BaseModel):
    query: str
    dates: Optional[List[date]] = None
    sources: Optional[List[str]] = None
    n: Optional[int] = 10


class SearchResult(BaseModel):
    text: str
    link: str


class SearchAndRerankRequest(BaseModel):
    query: str
    dates: Optional[List[date]] = None
    sources: Optional[List[str]] = None
    n_big: int = 20
    n_small: int = 5


def search_chroma_core(query: str, dates: Optional[List[date]], sources: Optional[List[str]], n: int) -> List[SearchResult]:
    processed_text = preprocess_text(query)
    query_texts = [processed_text]
    embeddings = voyage_embeddings_function(query_texts)

    filters = []

    if dates and len(dates) > 0:
        date_strs = [d.isoformat() for d in dates]
        filters.append({"date": {"$in": date_strs}})

    if sources and len(sources) > 0:
        filters.append({"source": {"$in": sources}})

    if len(filters) == 0:
        where_filter = None
    elif len(filters) == 1:
        where_filter = filters[0]
    else:
        where_filter = {"$and": filters}

    results = collection.query(
        query_embeddings=embeddings,
        include=["metadatas", "documents"],
        where=where_filter,
        n_results=n
    )

    final_results = []
    metadatas = results.get("metadatas", [[]])

    for meta in metadatas[0]:
        final_results.append(SearchResult(
            text=meta.get("text", ""),
            link=meta.get("id", ""),
        ))

    return final_results


def search_bm_25_core(query: str, dates: Optional[List[date]], sources: Optional[List[str]], n: int) -> List[SearchResult]:
    processed_query = preprocess_text(query)
    query_tokens = bm25s.tokenize(processed_query)

    if dates and len(dates) > 0:
        parquet_files = [os.path.join(base_dir, f"{d}.parquet") for d in dates if os.path.exists(os.path.join(base_dir, f"{d}.parquet"))]
    else:
        parquet_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".parquet")]

    if not parquet_files:
        return []

    df_list = []
    for pf in parquet_files:
        if os.path.exists(pf):
            df_current = pl.read_parquet(pf)
            df_list.append(df_current)

    if not df_list:
        return []

    df_all = pl.concat(df_list, how="vertical").unique(subset=["link"])

    if sources and len(sources) > 0:
        df_all = df_all.filter(pl.col("source").is_in(sources))

    if df_all.height == 0:
        return []

    if "text_cleaned" not in df_all.columns:
        df_all = df_all.with_columns(pl.col("text").alias("text_cleaned"))
    else:
        df_all = df_all.with_columns(
            pl.when(pl.col("text_cleaned").is_null())
            .then(pl.col("text"))
            .otherwise(pl.col("text_cleaned"))
            .alias("text_cleaned")
        )

    docs = df_all.to_dicts()
    corpus_text = [doc["text_cleaned"] for doc in docs]

    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus_text))

    top_k = n if n else 10
    results, scores = retriever.retrieve(query_tokens, k=top_k)

    result_indices = results[0]
    result_scores = scores[0]

    filtered = [(idx, sc) for idx, sc in zip(result_indices, result_scores) if sc > 0]

    filtered.sort(key=lambda x: x[1], reverse=True)

    final = []
    for idx, _ in filtered:
        doc = docs[int(idx)]
        final.append(SearchResult(
            text=doc.get("text", ""),
            link=doc.get("link", ""),
        ))

    return final


@router.post("/search_chroma", response_model=List[SearchResult])
async def search_chroma_documents(body: SearchRequest):
    return search_chroma_core(
        query=body.query,
        dates=body.dates,
        sources=body.sources,
        n=body.n or 10
    )


@router.post("/search_bm_25", response_model=List[SearchResult])
async def search_bm_25_documents(body: SearchRequest):
    return search_bm_25_core(
        query=body.query,
        dates=body.dates,
        sources=body.sources,
        n=body.n or 10
    )


def combine_and_deduplicate_results(chroma_results: List[SearchResult], bm25_results: List[SearchResult]) -> List[SearchResult]:
    combined = chroma_results + bm25_results
    seen_links = set()
    deduped = []
    for res in combined:
        if res.link not in seen_links:
            seen_links.add(res.link)
            deduped.append(res)
    return deduped


@router.post("/search", response_model=List[SearchResult])
async def combined_search(body: SearchRequest):
    chroma_results = search_chroma_core(
        query=body.query,
        dates=body.dates,
        sources=body.sources,
        n=body.n or 10
    )

    bm25_results = search_bm_25_core(
        query=body.query,
        dates=body.dates,
        sources=body.sources,
        n=body.n or 10
    )

    deduped = combine_and_deduplicate_results(chroma_results, bm25_results)

    return deduped


@router.post("/search_and_rerank", response_model=List[SearchResult])
async def search_and_rerank_documents(body: SearchAndRerankRequest):
    chroma_results = search_chroma_core(
        query=body.query,
        dates=body.dates,
        sources=body.sources,
        n=body.n_big
    )

    bm25_results = search_bm_25_core(
        query=body.query,
        dates=body.dates,
        sources=body.sources,
        n=body.n_big
    )

    deduped = combine_and_deduplicate_results(chroma_results, bm25_results)

    if not deduped:
        return []

    docs = [Document(page_content=res.text, metadata={"link": res.link}) for res in deduped]

    compressor = VoyageAIRerank(
        model=model_config.voyage_rerank_model_name,
        voyage_api_key=model_config.voyage_api_key,
        top_k=body.n_small
    )

    reranked_docs = compressor.compress_documents(docs, query=body.query)

    final_results = []
    for d in reranked_docs:
        link = d.metadata.get("link", "")
        text = d.page_content
        final_results.append(SearchResult(text=text, link=link))

    return final_results


@router.post("/answer", response_model=dict)
async def answer_documents(body: SearchAndRerankRequest):
    chroma_results = search_chroma_core(
        query=body.query,
        dates=body.dates,
        sources=body.sources,
        n=body.n_big
    )

    bm25_results = search_bm_25_core(
        query=body.query,
        dates=body.dates,
        sources=body.sources,
        n=body.n_big
    )

    deduped = combine_and_deduplicate_results(chroma_results, bm25_results)

    if not deduped:
        return {"answer": "По запросу ничего не найдено.", "results": []}

    docs = [Document(page_content=res.text, metadata={"link": res.link}) for res in deduped]

    compressor = VoyageAIRerank(
        model="rerank-2",
        voyage_api_key='pa--MJgY7dk_KiiGoU5iqFiwkNs6RyWAc7ElUsc6WJfCn8',
        top_k=body.n_small
    )

    reranked_docs = compressor.compress_documents(docs, query=body.query)

    if not reranked_docs:
        return {"answer": "По запросу ничего не найдено.", "results": []}

    llm = ChatOpenAI(
        model=model_config.openai_model_name,
        temperature=0,
        openai_api_key=model_config.openai_api_key
    )

    prompt_template = """Вы — помощник по анализу новостей. Вам предоставлен запрос пользователя и список новостей (заголовков или текстов) с метаданными. Ваша задача — найти и вывести только те новости из списка, которые явно упоминают персону, событие или тему, заданные в запросе. Если возможно, укажите ссылку на источник (из метаданных новости). Если ни одна новость не соответствует запросу, просто ответьте: «По запросу ничего не найдено.»    
Условия:
Используйте только факты, явно представленные в предоставленных новостях.
Не придумывайте дополнительную информацию.
Не давайте собственных комментариев или оценок, не делайте выводов, отсутствующих в источниках.
Если соответствующей информации нет, сообщите об этом без лишних подробностей.

Запрос пользователя: {query}

Новости: {context}

Ваш ответ:
"""
    template = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "context"]
    )

    llm_chain = LLMChain(llm=llm, prompt=template)

    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
    )

    result = chain.run(input_documents=reranked_docs, query=body.query)

    final_results = []
    for d in reranked_docs:
        link = d.metadata.get("link", "")
        text = d.page_content
        final_results.append(SearchResult(text=text, link=link))

    return {"answer": result, "results": [r.dict() for r in final_results]}

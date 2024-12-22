import os
from datetime import date
from typing import List, Optional

import bm25s
import polars as pl
from api.preprocess import preprocess_text, preprocess_text_b25
from api.parser import parse_rss, parse_all_sources, SOURCE, source_mapping
from api.models import hf_embeddings_function
from api.schema import SearchResult

import chromadb


class Storage:
    # TODO rewrite
    def __init__(self, max_changes: int = 1, host="localhost", port=8000):
        self.storage_path = f"{os.getcwd()}/app_data/news"
        self.count_changes = 0
        self.max_changes = max_changes
        self.chroma_client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.chroma_client.get_or_create_collection(
            "news",
            embedding_function=hf_embeddings_function,
            metadata={"hnsw:space": "cosine"},
        )
        os.makedirs(self.storage_path, exist_ok=True)
        self.df = pl.DataFrame()
        if os.path.exists(f"{self.storage_path}/data.parquet"):
            self.df = pl.read_parquet(
                f"{self.storage_path}/data.parquet"
            )  # да надо по нормальному но мне лень

    def embed_documents_to_collection(self, documents, embedding_function):
        ids = [doc["metadata"]["id"] for doc in documents]

        if ids:
            existing_documents = self.collection.get(ids=ids, include=["metadatas"])[
                "metadatas"
            ]
        else:
            existing_documents = []

        existing_ids = {
            metadata["id"] for metadata in existing_documents if metadata is not None
        }

        new_documents = [
            doc for doc in documents if doc["metadata"]["id"] not in existing_ids
        ]
        if not new_documents:
            return

        ids_new = [doc["metadata"]["id"] for doc in new_documents]
        documents_new = [doc["page_content"] for doc in new_documents]
        metadatas_new = [doc["metadata"] for doc in new_documents]

        embeddings = embedding_function(documents_new)

        self.collection.add(
            ids=ids_new,
            documents=documents_new,
            metadatas=metadatas_new,
            embeddings=embeddings,
        )

    def get_rss(self, source: SOURCE):
        self.count_changes += 1
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
                pl.col("published")
                .str.strptime(pl.Datetime, "%a, %d %b %Y %H:%M:%S %z", strict=False)
                .alias("published")
            )
            .with_columns(
                pl.col("published").cast(pl.Date).alias("date"),
                pl.col("text")
                .map_elements(preprocess_text, return_dtype=pl.Utf8)
                .alias("text_cleaned"),
                pl.col("text")
                .map_elements(preprocess_text_b25, return_dtype=pl.Utf8)
                .alias("text_cleaned_b25"),
            )
        )
        if self.df.is_empty():
            self.df = pl.concat([self.df, df]).unique(subset=["link"])
        else:
            self.df = df.unique(subset=["link"])

        if self.count_changes >= self.max_changes:
            self.df.write_parquet(f"{self.storage_path}/data.parquet")
            self.count_changes = 0

        df_dicts = self.df.to_dicts()
        docs = []
        for row in df_dicts:
            docs.append(
                {
                    "page_content": row.get("text_cleaned", "") or "",
                    "metadata": {
                        "id": row.get("link", ""),
                        "date": str(row.get("date", "")),
                        "source": row.get("source", ""),
                        "text": row.get("text", ""),
                    },
                }
            )
        self.embed_documents_to_collection(
            documents=docs, embedding_function=hf_embeddings_function
        )
        return parsed_entries

    def get_available_dates(self):
        if self.df.is_empty():
            return []
        return self.df["date"].unique().sort().to_list()

    def search_chroma_core(
        self,
        query: str,
        dates: Optional[List[date]],
        sources: Optional[List[str]],
        n: int,
    ) -> List[SearchResult]:
        if self.df.is_empty():
            return []

        processed_text = preprocess_text(query)
        query_texts = [processed_text]
        embeddings = hf_embeddings_function(query_texts)

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
        print(where_filter)
        results = self.collection.query(
            query_embeddings=embeddings,
            include=["metadatas", "documents"],
            where=where_filter,
            n_results=n,
        )

        final_results = []
        metadatas = results.get("metadatas", [[]])

        for meta in metadatas[0]:
            final_results.append(
                SearchResult(
                    text=meta.get("text", ""),
                    link=meta.get("id", ""),
                )
            )

        return final_results

    def search_bm_25_core(
        self,
        query: str,
        dates: Optional[List[date]],
        sources: Optional[List[str]],
        top_k: int = 10,
    ) -> List[SearchResult]:
        if self.df.is_empty():
            return []

        processed_query = preprocess_text_b25(query)
        query_tokens = bm25s.tokenize(processed_query)

        df_all = self.df
        if sources and len(sources) > 0:
            df_all = df_all.filter(pl.col("source").is_in(sources))

        if df_all.height == 0:
            return []

        docs = df_all.to_dicts()
        corpus_text = [doc["text_cleaned_b25"] for doc in docs]

        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(corpus_text))

        results, scores = retriever.retrieve(query_tokens, k=top_k)

        result_indices = results[0]
        result_scores = scores[0]

        filtered = [
            (idx, sc) for idx, sc in zip(result_indices, result_scores) if sc > 0
        ]

        filtered.sort(key=lambda x: x[1], reverse=True)

        final = []
        for idx, _ in filtered:
            doc = docs[int(idx)]
            final.append(
                SearchResult(
                    text=doc.get("text", ""),
                    link=doc.get("link", ""),
                )
            )

        return final


storage = Storage()

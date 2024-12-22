from datetime import date
from typing import List

from fastapi import APIRouter
from langchain.schema import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker


from api.schema import SearchResult, SearchRequest, SearchAndRerankRequest, Answer
from api.parser import SOURCE
from api.storages import storage
from api.models import rerank_model, chain


router = APIRouter()


@router.get("/rss/{source}", response_model=List[dict])
async def get_rss(source: SOURCE):
    return storage.get_rss(source=source)


@router.get("/available_dates", response_model=List[date])
async def get_available_dates():
    return storage.get_available_dates()


@router.post("/search_chroma", response_model=List[SearchResult])
async def search_chroma_documents(body: SearchRequest):
    return storage.search_chroma_core(
        query=body.query, dates=body.dates, sources=body.sources, n=body.n or 10
    )


@router.post("/search_bm_25", response_model=List[SearchResult])
async def search_bm_25_documents(body: SearchRequest):
    return storage.search_bm_25_core(
        query=body.query, dates=body.dates, sources=body.sources, top_k=body.n or 10
    )


def combine_and_deduplicate_results(
    chroma_results: List[SearchResult], bm25_results: List[SearchResult]
) -> List[SearchResult]:
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
    chroma_results = storage.search_chroma_core(
        query=body.query, dates=body.dates, sources=body.sources, n=body.n or 10
    )

    bm25_results = storage.search_bm_25_core(
        query=body.query,
        dates=body.dates,
        sources=body.sources,
        top_k=body.n,
    )

    deduped = combine_and_deduplicate_results(chroma_results, bm25_results)

    return deduped


@router.post("/search_and_rerank", response_model=List[SearchResult])
async def search_and_rerank_documents(body: SearchAndRerankRequest):
    chroma_results = storage.search_chroma_core(
        query=body.query, dates=body.dates, sources=body.sources, n=body.n_big
    )

    bm25_results = storage.search_bm_25_core(
        query=body.query, dates=body.dates, sources=body.sources, top_k=body.n_big
    )

    deduped = combine_and_deduplicate_results(chroma_results, bm25_results)

    if not deduped:
        return []

    docs = [
        Document(page_content=res.text, metadata={"link": res.link}) for res in deduped
    ]
    compressor = CrossEncoderReranker(model=rerank_model, top_n=body.n_small)

    reranked_docs = compressor.compress_documents(docs, query=body.query)

    final_results = []
    for d in reranked_docs:
        link = d.metadata.get("link", "")
        text = d.page_content
        final_results.append(SearchResult(text=text, link=link))

    return final_results


@router.post("/answer", response_model=Answer)
async def answer_documents(body: SearchAndRerankRequest):
    chroma_results = storage.search_chroma_core(
        query=body.query, dates=body.dates, sources=body.sources, n=body.n_big
    )

    bm25_results = storage.search_bm_25_core(
        query=body.query, dates=body.dates, sources=body.sources, top_k=body.n_big
    )

    deduped = combine_and_deduplicate_results(chroma_results, bm25_results)

    if not deduped:
        return {"answer": "По запросу ничего не найдено.", "results": []}

    docs = [
        Document(page_content=res.text, metadata={"link": res.link}) for res in deduped
    ]
    compressor = CrossEncoderReranker(model=rerank_model, top_n=body.n_small)

    reranked_docs = compressor.compress_documents(docs, query=body.query)

    if not reranked_docs:
        return {"answer": "По запросу ничего не найдено.", "results": []}

    result = chain.run(input_documents=reranked_docs, query=body.query)

    final_results = []
    for d in reranked_docs:
        link = d.metadata.get("link", "")
        text = d.page_content
        final_results.append(SearchResult(text=text, link=link))

    return {"answer": result, "results": [r.dict() for r in final_results]}

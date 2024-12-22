from pydantic import BaseModel
from datetime import date
from typing import List, Optional


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


class Answer(BaseModel):
    answer: str
    results: list[SearchResult]

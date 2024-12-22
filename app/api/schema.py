from pydantic import BaseModel
from datetime import date
from typing import List, Optional
from api.parser import SOURCE


class SearchRequest(BaseModel):
    query: str
    dates: Optional[List[date]] = None
    sources: Optional[List[SOURCE]] = None
    n: Optional[int] = 10


class SearchResult(BaseModel):
    text: str
    link: str


class SearchAndRerankRequest(BaseModel):
    query: str
    dates: Optional[List[date]] = None
    sources: Optional[List[SOURCE]] = None
    n_big: int = 20
    n_small: int = 5


class Answer(BaseModel):
    answer: str
    results: list[SearchResult]

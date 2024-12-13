from .semantic_search import get_search_semantic
from .full_text_search import get_search_full_text
from enum import Enum
from typing import Union


class SEARCH_TYPE(Enum):
    FULL_TEXT = 'full_text'
    SEMANTIC = 'semantic'


def get_search(
    querys: Union[str, list[str]],
    corpus_title: list[str],
    corpus_text: list[str],
    top_k: int = 2,
    search_type: SEARCH_TYPE = SEARCH_TYPE.SEMANTIC
):
    if search_type == SEARCH_TYPE.SEMANTIC:
        return get_search_semantic(querys, corpus_title, corpus_text, top_k)
    return get_search_full_text(querys, corpus_title, corpus_text, top_k)

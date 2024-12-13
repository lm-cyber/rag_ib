import bm25s
import Stemmer
from typing import Union
from collections import defaultdict
import numpy as np
stemmer = Stemmer.Stemmer("russian")


def get_search_full_text(
        querys: Union[str, list[str]],
        corpus_title: list[str],
        corpus_text: list[str],
        top_k: int = 2
):
    query_tokens = bm25s.tokenize(querys, stemmer=stemmer)

    retriever_title = bm25s.BM25()
    retriever_title.index(bm25s.tokenize(corpus_title, stemmer=stemmer))

    retriever_text = bm25s.BM25()
    retriever_text.index(bm25s.tokenize(corpus_text, stemmer=stemmer))
    results_text, scores_text = retriever_text.retrieve(query_tokens, k=top_k)
    results_title, scores_title = retriever_title.retrieve(
        query_tokens, k=top_k)

    dict_score = defaultdict(list)
    for index, score in zip(
        np.concatenate([results_text[0], results_title[0]]).tolist(),
        np.concatenate([scores_text[0], scores_title[0]]).tolist()
    ):
        dict_score[int(index)].append(float(score))

    for k in dict_score.keys():
        dict_score[k] = sum(dict_score[k])/len(dict_score[k])
    result = []
    for doc_id, score in sorted(dict_score.items(), key=lambda x: x[1]):
        result.append(
            {
                "title": corpus_title[doc_id],
                "text": corpus_text[doc_id],
                "score": score
            }
        )
    return result

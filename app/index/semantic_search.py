from typing import Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")


def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    return 1 - dot_product / (vec1 * vec2)


def get_embs(texts):
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    with torch.inference_mode():
        model_output = model(**encoded_input)
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)


def cosine_distance_search(query_vector, dataset):
    distances = [
        (idx, cosine_distance(query_vector, data_vector))
        for idx, data_vector in enumerate(dataset)
    ]
    distances.sort(key=lambda x: x[1])
    return distances


def get_search_semantic(
        querys: Union[str, list[str]],
        corpus_title: list[str],
        corpus_text: list[str],
        top_k: int = 2
):

    query_embs = get_embs([querys])[0].numpy()
    all_text = [f"{title}\n\n{text}" for title,
                text in zip(corpus_title, corpus_text)]
    texts_emb = get_embs(all_text).numpy()

    result_index = cosine_distance_search(query_embs, texts_emb)[top_k]
    result = []
    for doc_id, score in result_index:
        result.append(
            {
                "title": corpus_title[doc_id],
                "text": corpus_text[doc_id],
                "score": score
            }
        )
    return result

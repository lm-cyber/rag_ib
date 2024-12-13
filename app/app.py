import openai
import asyncio
from parser import parse_by_source, SOURCE, get_avaleble_date
from index import get_search, SEARCH_TYPE
from datetime import date
from shema import SearchResult, QueryResult
import openai
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from typing import Optional, Literal, Union
load_dotenv()
app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat_gpt_request(query, context):  # TODO make async
    prompt = f"""
          Вот некоторый соответствующий контекст:\n
          {context}\n
          Учитывая этот контекст, пожалуйста, ответьте на следующий вопрос:\n
          {query}"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


@app.get("/data")
async def get_data(
    query: str,
    date_search: Optional[date] = None,
    source: Optional[Union[Literal[SOURCE.LENTA, SOURCE.RBC,
                                   SOURCE.ALL, SOURCE.DUMP], str]] = SOURCE.RBC,
    search_type: Optional[Union[Literal[SEARCH_TYPE.SEMANTIC,
                                        SEARCH_TYPE.FULL_TEXT], str]] = SEARCH_TYPE.FULL_TEXT,
    top_k: int = 2
):
    corpus = await parse_by_source(date_search, source)
    result = get_search(query, corpus['title'],
                        corpus['text'], top_k, search_type)
    sum_of_context = '\n'.join(
        list(map(lambda x: f"{x['title']}\n\n{x['text']}", result))
    )
    return SearchResult(
        summyry=chat_gpt_request(query, sum_of_context),
        date=date_search,
        query_result=list(map(lambda x: QueryResult(**x), result))
    )


@ app.get('/get_date')
async def get_date(source=SOURCE.DUMP):
    return get_avaleble_date()  # TODO

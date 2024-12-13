import httpx
import asyncio
from collections import deque
import feedparser


def get_parser(rss_link):
    async def rss_parser():
        httpx_client = httpx.AsyncClient()

        response = await httpx_client.get(rss_link)
        feed = feedparser.parse(response.text)
        sums = []
        titles = []

        for entry in feed.entries[::-1]:
            sums.append(entry['summary'] if len(
                entry['summary']) != 0 else "нету текста")
            titles.append(entry['title'])
        return {
            'title': titles,
            'text': sums
        }
    return rss_parser

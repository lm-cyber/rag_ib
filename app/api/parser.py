import feedparser
from enum import Enum


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
        summary = (
            entry.get("summary", "")
            or entry.get("description", "")
            or entry.get("value", "")
        )
        if summary:
            title = title + " " + summary
        parsed_entries.append(
            {
                "text": title,
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "source": source,
            }
        )
    return parsed_entries


def parse_all_sources():
    all_entries = []
    for s, url in source_mapping.items():
        if s != SOURCE.ALL:
            all_entries.extend(parse_rss(url, s))
    return all_entries

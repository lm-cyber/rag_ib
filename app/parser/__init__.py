from .dump_parse import get_by_date_dump, get_all_data
from enum import Enum
from .parser import get_parser

# TODO add more


class SOURCE(Enum):
    DUMP = "dump"
    LENTA = 'lenta'
    RBC = 'rbc'
    ALL = 'all'


source_mapping = {
    SOURCE.LENTA: "https://lenta.ru/rss/news",
    SOURCE.RBC: 'https://rssexport.rbc.ru/rbcnews/news/20/full.rss',

}


async def parse_by_source(date_str=None, source=SOURCE.DUMP):
    source = SOURCE(source) if isinstance(source, str)else source
    if source == SOURCE.DUMP:
        return get_by_date_dump(date_str)
    return (await {
        SOURCE.ALL: parse_from_all,
        SOURCE.LENTA: get_parser(source_mapping[source]),
        SOURCE.RBC: get_parser(source_mapping[source]),
    }[source]())


def get_avaleble_date(source=SOURCE.DUMP):
    return get_all_data()  # TODO


async def parse_from_all():
    titles = []
    sums = []
    for v in source_mapping.values():
        title, summ = await get_parser(v)()
        titles.extend(title)
        sums.extend(summ)
    return {
        'title': titles,
        'text': sums
    }

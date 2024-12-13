from datetime import date
import polars as pl
data = pl.read_csv('data.csv')
data = data.with_columns(pl.col('date').str.to_date())
dates = data['date'].unique().to_list()


def get_by_date_dump(date_conv) -> dict[str, list[str]]:
    select_by_date = data.filter(
        pl.col('date') == date_conv)

    return {
        'title': select_by_date['title'].to_list(),
        'text': select_by_date['text'].to_list()
    }


def get_all_data():
    return dates

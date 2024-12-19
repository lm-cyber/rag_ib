from cleantext import clean
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from spacy import load

stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")
nlp = load("ru_core_news_sm")

args = {
    'fix_unicode': True,
    'to_ascii': False,
    'lower': True,
    'normalize_whitespace': True,
    'no_line_breaks': True,
    'strip_lines': True,
    'keep_two_line_breaks': False,
    'no_urls': True,
    'no_emails': True,
    'no_phone_numbers': True,
    'no_numbers': False,
    'no_digits': False,
    'no_currency_symbols': True,
    'no_punct': True,
    'no_emoji': True,
    'replace_with_url': "<ссылка>",
    'replace_with_email': "<почта>",
    'replace_with_phone_number': "<телефон>",
    'replace_with_number': "",
    'replace_with_digit': "",
    'replace_with_currency_symbol': "<валюта>",
    'replace_with_punct': "",
    'lang': "ru",
}


def preprocess_text(text):
    text = clean(text, **args)
    words = word_tokenize(text, language="russian")
    words = [word for word in words if word not in stop_words]
    doc = nlp(' '.join(words))
    lemmatized_words = [token.lemma_ for token in doc]
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    return ' '.join(stemmed_words)

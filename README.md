# News Aggregator with LLM-Powered Search

Данный проект представляет собой агрегатор новостей с возможностью семантического и классического поиска, переранжировки
результатов и генерации итоговых ответов с помощью LLM-моделей. Проект автоматизирует сбор новостей из заданных
RSS-источников, последующую индексацию, а также интеграцию со стеком LangChain + Chromadb + VoyageAI Embeddings для
реализации продвинутых сценариев поиска и анализа.

## Основные возможности

1. **Сбор новостей из RSS:**  
   Поддерживаются несколько популярных русскоязычных новостных ресурсов (Лента.ру, РБК, ТАСС, РИА Новости, Ведомости).
   Новости скачиваются, предварительно обрабатываются и сохраняются в формате parquet.

2. **Преобразование и обработка данных:**
    - Очистка текстов от лишних символов.
    - Хранение данных по датам.
    - Индексация новостных документов в Chromadb с использованием VoyageAI Embeddings для дальнейшего семантического
      поиска.

3. **Семантический поиск (Chroma + Embeddings):**  
   Возможность искать документы по смыслу запроса, а не просто по ключевым словам.

4. **Классический поиск (BM25):**  
   Для сравнения результатов можно использовать классический поиск по терминам.

5. **Комбинированный поиск:**  
   Объединение результатов из Chroma (семантика) и BM25 (термы) с последующей дедупликацией.

6. **Переранжировка результатов (Rerank):**  
   С помощью VoyageAIRerank модели можно улучшить порядок выдачи, выбирая самые релевантные документы.

7. **Генерация итогового ответа (Answer Endpoint):**  
   Использование ChatOpenAI (LangChain chain) для формирования итогового ответа на основе выбранных документов. Запрос
   может быть:
    - «Какие сегодня есть события (новости), упоминающие ФИО?»
    - «Выведи полный текст новостей о [персоне/событии]»

   Итоговый ответ формируется без выдумок и содержит только фактическую информацию из найденных источников.

8. **Автоматизация процесса:**  
   Вся логика (от загрузки новостей до получения ответа от LLM) может быть выстроена в автоматическом pipeline,
   доступном через REST API.

## Технологический стек

- **Язык:** Python
- **Фреймворк:** FastAPI
- **База данных векторов:** Chromadb
- **LLM и эмбеддинги:** LangChain, VoyageAI Embeddings, ChatOpenAI
- **Классический поиск:** BM25
- **Хранилище данных:** Parquet-файлы

## Доступные эндпоинты

Все эндпоинты доступны под роутером `APIRouter`.

### 1. Загрузка новостей из RSS

**GET** `/rss/{source}`  
Позволяет загрузить новости из выбранного источника. Параметр `{source}` может быть: `lenta`, `rbc`, `tass`, `ria`,
`vedomosti`, `all`.

**Пример:**  
`GET /rss/lenta` – загрузка новостей с Ленты.ру.

**Ответ:** JSON со списком новостей, содержащих поля `text`, `link`, `published`, `source`.

### 2. Семантический поиск (Chroma)

**POST** `/search_chroma`  
Делает поиск по векторному хранилищу с использованием VoyageAI Embeddings.

**Формат запроса (JSON):**

```json
{
  "query": "Запрос пользователя",
  "dates": [
    "2024-12-19"
  ],
  "sources": [
    "rbc",
    "tass"
  ],
  "n": 10
}
```

**Ответ:** список объектов, каждый из которых содержит `text` и `link` найденных документов.

### 3. Классический поиск (BM25)

**POST** `/search_bm_25`  
Делает поиск по ключевым словам с использованием BM25.

**Формат запроса и ответ:** аналогичен `/search_chroma`.

### 4. Комбинированный поиск

**POST** `/search`  
Объединяет результаты из Chroma и BM25, устраняя дубли.

**Формат запроса и ответ:** аналогичен вышеописанным эндпоинтам.

### 5. Поиск с последующей переранжировкой

**POST** `/search_and_rerank`  
Сначала ищет расширенный список результатов (`n_big`), а затем с помощью VoyageAIRerank отбирает топ-результаты (
`n_small`).

**Формат запроса (JSON):**

```json
{
  "query": "Запрос пользователя",
  "dates": [
    "2024-12-19"
  ],
  "sources": [
    "rbc"
  ],
  "n_big": 20,
  "n_small": 5
}
```

**Ответ:** Список из `n_small` наилучших результатов после переранжировки.

### 6. Генерация итогового ответа (LLM Answer)

**POST** `/answer`  
Выполняет поиск, затем переранжировку, а затем использует LLM для создания итогового ответа, учитывая факты из найденных
документов.

**Формат запроса (JSON):** аналогичен `/search_and_rerank`.

**Ответ:**

```json
{
  "answer": "Сформированный итоговый ответ",
  "results": [
    {
      "text": "Текст новости",
      "link": "URL"
    },
  ]
}
```

### 7. Доступные даты

**GET** `/available_dates`  
Возвращает список дат, за которые есть сохранённые данные.

**Ответ:** список дат в ISO формате (YYYY-MM-DD).

## Запуск проекта

```bash
docker compose -f docker-compose.yaml -p rag_ib up -d --build
```
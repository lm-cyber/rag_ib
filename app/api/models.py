from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from chromadb import EmbeddingFunction, Documents, Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
import os


class HFEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        model_name = "cointegrated/LaBSE-en-ru"
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, encode_kwargs=encode_kwargs
        )

    def __call__(self, input: Documents) -> Embeddings:
        return self.embeddings.embed_documents(texts=input)

    def embed_query(self, query: str) -> Embeddings:
        return self.embeddings.embed_documents([query])


rerank_model = HuggingFaceCrossEncoder(model_name="qilowoq/bge-reranker-v2-m3-en-ru")


def get_chain():
    prompt_template = """Вы — помощник по анализу новостей. Вам предоставлен запрос пользователя и список новостей (заголовков или текстов) с метаданными. Ваша задача — найти и вывести только те новости из списка, которые явно упоминают персону, событие или тему, заданные в запросе. Если возможно, укажите ссылку на источник (из метаданных новости). Если ни одна новость не соответствует запросу, просто ответьте: «По запросу ничего не найдено.»    
Условия:
Используйте только факты, явно представленные в предоставленных новостях.
Не придумывайте дополнительную информацию.
Не давайте собственных комментариев или оценок, не делайте выводов, отсутствующих в источниках.
Если соответствующей информации нет, сообщите об этом без лишних подробностей.

Запрос пользователя: {query}

Новости: {context}

Ваш ответ:
"""

    template = PromptTemplate(
        template=prompt_template, input_variables=["query", "context"]
    )
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    low_memory = os.getenv("LOWMEM") == "true"
    if low_memory:
        print("Low memory mode\n\n\n\n\n\n\n\n")
        model_file = hf_hub_download(
            repo_id="IlyaGusev/saiga_llama3_8b_gguf", filename="model-q2_K.gguf"
        )
    else:
        print("Normal memory mode\n\n\n\n\n\n\n\n")
        model_file = hf_hub_download(
            repo_id="t-tech/T-lite-it-1.0-Q8_0-GGUF", filename="t-lite-it-1.0-q8_0.gguf"
        )

    llm = LlamaCpp(
        model_path=model_file,
        n_gpu_layers=-1,
        n_batch=1024,
        callback_manager=callback_manager,
        verbose=True,
        n_ctx=1024,
    )

    llm_chain = LLMChain(llm=llm, prompt=template)

    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
    )
    return chain


chain = get_chain()
hf_embeddings_function = HFEmbeddingFunction()

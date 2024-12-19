from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    openai_api_key: str
    openai_model_name: str
    voyage_api_key: str
    voyage_embeddings_model_name: str
    voyage_rerank_model_name: str

    model_config = SettingsConfigDict(env_prefix='MODEL_')


model_config = ModelConfig()
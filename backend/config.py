from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    typhoon_api_key: str
    typhoon_base_url: str = "https://api.opentyphoon.ai/v1"
    typhoon_model: str = "typhoon-v2.5-30b-a3b-instruct"
    embedding_model: str = "intfloat/multilingual-e5-large"
    chroma_persist_dir: str = "./chroma_db"
    kb_dir: str = "./backend/knowledge_base"
    top_k: int = 5

    branch_ids: list[str] = ["branch_main", "branch_2", "branch_fullbody"]


settings = Settings()

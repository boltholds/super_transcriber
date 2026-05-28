from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_token: str

    whisperx_model: str = "small"
    whisperx_device: str = "auto"
    whisperx_compute_type: str = "auto"

    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "local"
    llm_model: str = "qwen2.5-7b-instruct"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "YZZ Backend"
    API_V1_STR: str = "/api/v1"

    # LLM Settings
    OPENAI_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    VOLCENGINE_API_KEY: str | None = None
    
    DEFAULT_LLM_PROVIDER: str = "openai"
    
    class Config:
        env_file = ".env"

settings = Settings()

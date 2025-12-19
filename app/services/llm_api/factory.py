from app.services.llm_api.base import LLMProvider
from app.core.config import settings

class LLMServiceFactory:
    @staticmethod
    def get_provider(provider_name: str = None) -> LLMProvider:
        provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER

        raise ValueError(f"Unsupported LLM provider: {provider_name}")

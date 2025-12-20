from app.services.llm_api.base import LLMProvider
from app.services.llm_api.byteDance.provider import ByteDanceProvider
from app.services.llm_api.openai.provider import OpenAIProvider
from app.services.llm_api.deepseek.provider import DeepSeekProvider
from app.core.config import settings

class LLMServiceFactory:
    @staticmethod
    def get_provider(provider_name: str = None) -> LLMProvider:
        provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER
        
        if provider_name == "volcengine" or provider_name == "bytedance":
            return ByteDanceProvider()
        elif provider_name == "openai":
            return OpenAIProvider()
        elif provider_name == "deepseek":
            return DeepSeekProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

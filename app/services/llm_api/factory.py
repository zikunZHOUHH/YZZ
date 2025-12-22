from app.services.llm_api.base import LLMProvider
from app.services.llm_api.byteDance.provider import ByteDanceProvider
from app.services.llm_api.openai.provider import OpenAIProvider
from app.services.llm_api.deepseek.provider import DeepSeekProvider
from app.core.config import settings

class LLMServiceFactory:
    """
    用于实例化相应 LLM 提供商的工厂类。
    """
    @staticmethod
    def get_provider(provider_name: str = None) -> LLMProvider:
        """
        获取指定 LLM 提供商的实例。
        
        Args:
            provider_name (str, optional): 提供商名称（例如 'openai', 'deepseek'）。
                                           默认为配置的默认提供商。
                                           
        Returns:
            LLMProvider: 请求的提供商实例。
            
        Raises:
            ValueError: 如果不支持该提供商名称。
        """
        provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER
        
        if provider_name == "volcengine" or provider_name == "bytedance":
            return ByteDanceProvider()
        elif provider_name == "openai":
            return OpenAIProvider()
        elif provider_name == "deepseek":
            return DeepSeekProvider()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

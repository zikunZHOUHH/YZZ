from abc import ABC, abstractmethod
from typing import AsyncGenerator
from app.schemas.llm import LLMRequest, LLMResponse

class LLMProvider(ABC):
    """
    LLM 提供商的抽象基类。
    所有具体的提供商实现都必须继承此类。
    """
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        从 LLM 生成完整的响应。
        
        Args:
            request (LLMRequest): 请求参数。
            
        Returns:
            LLMResponse: 完整的生成响应。
        """
        pass

    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """
        从 LLM 生成流式响应。
        
        Args:
            request (LLMRequest): 请求参数。
            
        Yields:
            str: 生成的文本内容分块。
        """
        pass

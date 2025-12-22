from abc import ABC, abstractmethod
from typing import AsyncGenerator
from app.schemas.llm import LLMRequest, LLMResponse

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    All specific provider implementations must inherit from this class.
    """
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a complete response from the LLM.
        
        Args:
            request (LLMRequest): The request parameters.
            
        Returns:
            LLMResponse: The complete generated response.
        """
        pass

    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            request (LLMRequest): The request parameters.
            
        Yields:
            str: Chunks of generated text content.
        """
        pass

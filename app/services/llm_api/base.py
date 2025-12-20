from abc import ABC, abstractmethod
from typing import AsyncGenerator
from app.schemas.llm import LLMRequest, LLMResponse

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a complete response"""
        pass

    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate a streaming response"""
        pass

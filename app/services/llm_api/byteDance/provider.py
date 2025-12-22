from app.services.llm_api.base import LLMProvider
from app.schemas.llm import LLMRequest, LLMResponse
from app.core.config import settings
from volcenginesdkarkruntime import Ark
import asyncio
from functools import partial
from typing import AsyncGenerator

class ByteDanceProvider(LLMProvider):
    def __init__(self):
        self.api_key = settings.VOLCENGINE_API_KEY
        if not self.api_key:
            # Allow instantiation even if key is missing, validation happens at call time
            pass
            
        if self.api_key:
            self.client = Ark(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=self.api_key
            )
        else:
            self.client = None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.client:
            raise ValueError("Volcengine (ByteDance) API key not configured")
            
        model = request.model
        if not model:
            raise ValueError("Model (Endpoint ID) is required for ByteDance provider")

        messages = [{"role": "user", "content": request.prompt}]

        # Use asyncio.to_thread (Python 3.9+) or loop.run_in_executor
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None,
            partial(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )
        )

        content = completion.choices[0].message.content
        usage = completion.usage.model_dump() if completion.usage else None

        return LLMResponse(
            content=content,
            provider="bytedance",
            model=model,
            usage=usage
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using ByteDance's Ark API.
        Executes the blocking SDK call in a separate thread to support async streaming.
        
        Args:
            request (LLMRequest): The request parameters.
            
        Yields:
            str: Real-time chunks of generated text content.
            
        Raises:
            ValueError: If the API key or model (Endpoint ID) is missing.
            RuntimeError: If the API call fails.
        """
        if not self.client:
            raise ValueError("Volcengine (ByteDance) API key not configured")
            
        model = request.model
        if not model:
            raise ValueError("Model (Endpoint ID) is required for ByteDance provider")

        messages = [{"role": "user", "content": request.prompt}]
        loop = asyncio.get_running_loop()

        try:
            # Create the stream generator in a thread to avoid blocking
            stream = await loop.run_in_executor(
                None,
                partial(
                    self.client.chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=True
                )
            )

            # Iterate over the stream in a thread
            while True:
                try:
                    chunk = await loop.run_in_executor(None, next, stream)
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                except StopIteration:
                    break
        except Exception as e:
            raise RuntimeError(f"ByteDance API Stream Error: {str(e)}")

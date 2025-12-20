from app.services.llm_api.base import LLMProvider
from app.schemas.llm import LLMRequest, LLMResponse
from app.core.config import settings
from openai import AsyncOpenAI
from typing import AsyncGenerator

class DeepSeekProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        api_key = settings.DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError("DeepSeek API key not configured")
        
        # Initialize AsyncOpenAI client with DeepSeek base URL
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
            
        model = request.model or "deepseek-chat"
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": request.prompt}
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )
            
            content = response.choices[0].message.content
            # OpenAI object to dict or access directly
            usage = response.usage.model_dump() if response.usage else None
            
            return LLMResponse(
                content=content,
                provider="deepseek",
                model=model,
                usage=usage
            )
        except Exception as e:
            raise RuntimeError(f"DeepSeek API Error: {str(e)}")

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        api_key = settings.DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError("DeepSeek API key not configured")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
            
        model = request.model or "deepseek-chat"
        
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": request.prompt}
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise RuntimeError(f"DeepSeek API Stream Error: {str(e)}")

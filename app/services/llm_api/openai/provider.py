from app.services.llm_api.base import LLMProvider
from app.schemas.llm import LLMRequest, LLMResponse
from app.core.config import settings
from openai import AsyncOpenAI
from typing import AsyncGenerator

class OpenAIProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key not configured")
            
        # Initialize AsyncOpenAI client
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1" # Default is correct
        )
        
        model = request.model or "gpt-3.5-turbo"
        
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
            usage = response.usage.model_dump() if response.usage else None
            
            return LLMResponse(
                content=content,
                provider="openai",
                model=model,
                usage=usage
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API Error: {str(e)}")

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key not configured")
            
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"
        )
        
        model = request.model or "gpt-3.5-turbo"
        
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
            raise RuntimeError(f"OpenAI API Stream Error: {str(e)}")

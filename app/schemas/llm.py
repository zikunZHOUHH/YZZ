from pydantic import BaseModel
from typing import Optional, List, Any

class LLMRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: bool = False

class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: Optional[dict[str, Any]] = None

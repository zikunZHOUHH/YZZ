from fastapi import APIRouter, HTTPException, Depends
from app.schemas.llm import LLMRequest, LLMResponse
from app.services.llm.factory import LLMServiceFactory

router = APIRouter()

@router.post("/chat", response_model=LLMResponse)
async def chat_completion(request: LLMRequest):
    """
    Generate chat completion using the specified or default LLM provider.
    """
    try:
        service = LLMServiceFactory.get_provider(request.provider)
        response = await service.generate(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Provider Error: {str(e)}")

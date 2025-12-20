from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app.schemas.llm import LLMRequest, LLMResponse
from app.services.llm_api.factory import LLMServiceFactory

router = APIRouter()

@router.post("/chat")
async def chat_completion(request: LLMRequest):
    """
    Generate chat completion using the specified or default LLM provider.
    Supports both streaming and non-streaming responses based on request.stream flag.
    """
    try:
        service = LLMServiceFactory.get_provider(request.provider)
        
        if request.stream:
            return StreamingResponse(
                service.generate_stream(request),
                media_type="text/event-stream"
            )
        else:
            response = await service.generate(request)
            return response
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Provider Error: {str(e)}")

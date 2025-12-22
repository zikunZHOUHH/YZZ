from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app.schemas.llm import LLMRequest, LLMResponse
from app.services.llm_api.factory import LLMServiceFactory

router = APIRouter()

@router.post("/chat")
async def chat_completion(request: LLMRequest):
    """
    使用指定或默认的 LLM 提供商生成聊天回复。
    
    根据 request.stream 标志支持流式和非流式响应。
    
    Args:
        request (LLMRequest): 包含提示词、提供商、模型等信息的请求对象。
        
    Returns:
        LLMResponse | StreamingResponse: 生成的响应对象或 token 流。
        
    Raises:
        HTTPException: 如果提供商无效或发生 API 错误。
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

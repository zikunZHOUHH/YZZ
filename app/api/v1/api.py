from fastapi import APIRouter

from app.api.v1.endpoints import health, llm

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(llm.router, prefix="/llm_api", tags=["llm_api"])

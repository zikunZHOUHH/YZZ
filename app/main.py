from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.api import api_router

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    """
    根端点，用于验证 API 状态。
    
    Returns:
        dict: 包含欢迎信息和文档链接的字典。
    """
    return {"message": "Welcome to YZZ Backend", "docs_url": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

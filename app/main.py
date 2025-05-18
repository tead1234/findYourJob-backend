import os
import sys
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.controllers import user_face_controller
from fastapi.staticfiles import StaticFiles
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 미들웨어를 먼저 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# IP 제한 미들웨어
@app.middleware("http")
async def ip_restriction_middleware(request: Request, call_next):
    try:
        client_ip = request.client.host
        logger.debug(f"Request from IP: {client_ip}")
        
        # 모든 IP 허용 (개발 환경)
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error in IP restriction middleware: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 에러 핸들링 미들웨어
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Include the router from the controller module
app.include_router(user_face_controller.router, prefix='/face_api')
static_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_images")

# StaticFiles 경로를 절대 경로로 설정합니다.
app.mount("/ai_images", StaticFiles(directory=static_folder_path), name="static")

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
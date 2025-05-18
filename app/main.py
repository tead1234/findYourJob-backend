import os
import sys
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.controllers import user_face_controller
from fastapi.staticfiles import StaticFiles
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 허용된 IP 목록
ALLOWED_IPS = [
    "127.0.0.1",  # localhost
    "::1",        # localhost IPv6
    "152.53.248.175",  # 허용된 외부 IP
    # 여기에 추가로 허용할 IP를 넣으세요
]

app = FastAPI()

# IP 제한 미들웨어
@app.middleware("http")
async def ip_restriction_middleware(request: Request, call_next):
    client_ip = request.client.host
    if client_ip not in ALLOWED_IPS:
        raise HTTPException(status_code=403, detail="Access denied")
    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include the router from the controller module
app.include_router(user_face_controller.router, prefix='/face_api')
static_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_images")

# Include the router from the controller module
app.include_router(user_face_controller.router, prefix='/face_api')

# StaticFiles 경로를 절대 경로로 설정합니다.
app.mount("/ai_images", StaticFiles(directory=static_folder_path), name="static")

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
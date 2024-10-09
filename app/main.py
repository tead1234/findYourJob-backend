import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.controllers import user_face_controller
from fastapi.staticfiles import StaticFiles
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app = FastAPI()

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
app.mount("/static", StaticFiles(directory=static_folder_path), name="static")

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
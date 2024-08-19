import os
import sys

from fastapi import FastAPI
import uvicorn
from controllers import user_face_controller
from fastapi.staticfiles import StaticFiles
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app = FastAPI()

# Include the router from the controller module
app.include_router(user_face_controller.router, prefix='/face_api')
app.mount("/static", StaticFiles(directory="ai_images"), name="static")

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
from fastapi import FastAPI
import uvicorn
from controllers import image_controller

app = FastAPI()

# Include the router from the controller module
app.include_router(image_controller.router, prefix='/face_api/')


@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
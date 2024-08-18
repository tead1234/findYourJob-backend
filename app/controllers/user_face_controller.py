from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from bson import ObjectId
import io
import logging
from app.services.face_analysis_service import *

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        predict()  ## 유저
        이미지를
        학습에
        사용하는걸
        동의할지
        말지를
        알려준다.
        이미지를
        분석한
        정보를
        db에
        저장해야.
        repo에서
        받아온
        이미지를
        학습한
        face객체를
        만들어서
        저장.
        file_id = await process(file)
        return {"file_id": str(file_id), "filename": file.filename}
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images/{file_id}")
async def get_image(file_id: str):
    try:

        return await get_image_service(file_id)
    except Exception as e:
        logger.error(f"Error getting image: {e}")
        raise HTTPException(status_code=404, detail="File not found")



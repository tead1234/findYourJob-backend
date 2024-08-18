import io
import os

import dlib
import cv2
import numpy as np
from config import get_async_gridfs
from bson import ObjectId
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

# 모델 파일 경로
MODEL_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_FILE_PATH)


async def upload_image_service(file):
    gridfs, db = await get_async_gridfs()
    file_content = await file.read()
    file_id = await gridfs.upload_from_stream(file.filename, io.BytesIO(file_content))

    collection = db['fs.files']
    await collection.update_one(
        {"_id": file_id},
        {"$set": {"filename": file.filename, "contentType": file.content_type}}
    )
    return file_id


async def get_file_metadata_service(file_id):
    _, db = await get_async_gridfs()
    collection = db['fs.files']
    object_id = ObjectId(file_id)
    file_metadata = await collection.find_one({"_id": object_id})
    if not file_metadata:
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "file_id": str(file_metadata["_id"]),
        "filename": file_metadata["filename"],
        "length": file_metadata["length"],
        "upload_date": file_metadata["uploadDate"],
        "content_type": file_metadata["contentType"]
    }


async def get_image_service(file_id):
    gridfs, _ = await get_async_gridfs()
    file_metadata = await get_file_metadata_service(file_id)
    object_id = ObjectId(file_id)
    file_stream = await gridfs.open_download_stream(object_id)
    file_content = await file_stream.read()
    return StreamingResponse(io.BytesIO(file_content),
                             media_type=file_metadata.get("content_type", 'application/octet-stream'))


async def delete_image_service(file_id):
    gridfs, _ = await get_async_gridfs()
    object_id = ObjectId(file_id)
    await gridfs.delete(object_id)

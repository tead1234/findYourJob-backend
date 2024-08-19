# from fastapi import APIRouter, UploadFile, File, HTTPException
# from fastapi.responses import StreamingResponse
# from bson import ObjectId
# import io
# import logging
#
# from app.services.image_service import (
#     upload_image_service,
#     get_file_metadata_service,
#     get_image_service,
#     process_image_service,
#     delete_image_service
# )
#
# router = APIRouter()
# logger = logging.getLogger(__name__)
#
# @router.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     try:
#         file_id = await upload_image_service(file)
#         return {"file_id": str(file_id), "filename": file.filename}
#     except Exception as e:
#         logger.error(f"Error uploading image: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#
# @router.get("/file_metadata/{file_id}")
# async def get_file_metadata_route(file_id: str):
#     try:
#         file_metadata = await get_file_metadata_service(file_id)
#         return file_metadata
#     except Exception as e:
#         logger.error(f"Error getting metadata image: {e}")
#         raise HTTPException(status_code=404, detail="File not found")
#
# @router.get("/images/{file_id}")
# async def get_image(file_id: str):
#     try:
#         return await get_image_service(file_id)
#     except Exception as e:
#         logger.error(f"Error getting image: {e}")
#         raise HTTPException(status_code=404, detail="File not found")
#
# @router.get("/process_image/{file_id}")
# async def process_image(file_id: str):
#     try:
#         return await process_image_service(file_id)
#     except Exception as e:
#         logger.error(f"Error processing image: {e}")
#         raise HTTPException(status_code=404, detail="File not found")
#
# @router.delete("/images/{file_id}")
# async def delete_image(file_id: str):
#     try:
#         await delete_image_service(file_id)
#         return {"status": "File deleted"}
#     except Exception as e:
#         logger.error(f"Error deleting image: {e}")
#         raise HTTPException(status_code=404, detail="File not found")

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from bson import ObjectId
import io
import logging
from app.services.face_analysis_service import *
from app.services.machine_learning_service import *
from app.services.user_interface_service import *
from io import BytesIO

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # 파일이 이미지 검증
    if not is_image_file(file):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image type")

    try:
        file_content = await file.read()
        file_stream = BytesIO(file_content)

        # 얼굴 랜드마크 추출 및 Numpy 배열 반환
        landmarks_np = get_face_landmarks(file_stream)

        ## 동의 시 <- entity/face_image 형태로 저장함 함수는 face_image_repository/save_face_image

        # 예측 모델 호출
        prediction_service = machine_learning_service()
        prediction = prediction_service.predict(landmarks_np)

        prediction["predicted_job1_image"] = encode_image_to_base64(f"ai_images/{prediction['predicted_job1']}.jpg")
        prediction["predicted_job2_image"] = encode_image_to_base64(f"ai_images/{prediction['predicted_job2']}.jpg")
        prediction["predicted_job3_image"] = encode_image_to_base64(f"ai_images/{prediction['predicted_job3']}.jpg")

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @router.get("/images/{file_id}")
# async def get_image(file_id: str):
#     try:
#
#         return await get_image_service(file_id)
#     except Exception as e:
#         logger.error(f"Error getting image: {e}")
#         raise HTTPException(status_code=404, detail="File not found")



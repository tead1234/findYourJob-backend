from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from bson import ObjectId
import io
import uuid
import logging
from app.services.face_analysis_service import *
from app.services.machine_learning_service import *
from app.services.user_interface_service import *
from app.entity.face_image import face
from io import BytesIO
from app.repositories.face_image_repository import *

router = APIRouter()
logger = logging.getLogger(__name__)
# face_image_repo = face_image_repository()

@router.post("/upload/")
async def upload_image(file: UploadFile = File(...),
                       consent: bool = Query(..., description="유저 개인정보 동의"),
                       gender: str = Query(..., description="성별 (men/women)")
                       ):
    # 파일이 이미지 검증
    if not is_image_file(file):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image type")

    try:
        file_content = await file.read()
        file_stream = BytesIO(file_content)

        # 얼굴 랜드마크 추출 및 Numpy 배열 반환
        landmarks_np = get_face_landmarks(file_stream)
        landmarks_list = convert_landmarks_to_list(landmarks_np)

        # 예측 모델 호출
        prediction_service = machine_learning_service()
        prediction = prediction_service.predict(landmarks_np)

        # 동의시 학습을 위해 mongodb에 저장
        if consent:
            print("consent flow")
            face_entity = face(
                id=str(uuid.uuid4()),             # 랜덤 UUID 생성
                gender=gender,                  # 파라미터로 받은 gender
                landmarks=landmarks_list,         # numpy 배열을 리스트로 변환
                job1=prediction['predicted_job1_translated'],
                job2=prediction['predicted_job2_translated'],
                job3=prediction['predicted_job3_translated']
            )
            face_image_repo = face_image_repository()
            result_id = await face_image_repo.save_face_image(face_image=face_entity)
            picture_number = await face_image_repo.get_total_records()

            if picture_number > 50:
                print("picture_number",picture_number)
                machine_learning = machine_learning_service()
                await machine_learning.self_learn()
            else:
                print("50개가 넘지 않습니다. 학습하지 않습니다.")

            print(f"Saved face image with ID: {result_id}")

        prediction["predicted_job1_image"] = f"ai_images/{prediction['predicted_job1']}.jpg"
        prediction["predicted_job2_image"] = f"ai_images/{prediction['predicted_job2']}.jpg"
        prediction["predicted_job3_image"] = f"ai_images/{prediction['predicted_job3']}.jpg"

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# def train():
#     picture_number = face_image_repo.get_total_records()
#     if picture_number > 50:
#         machine_learning = machine_learning_service()
#         machine_learning.self_learn()
# @router.get("/images/{file_id}")
# async def get_image(file_id: str):
#     try:
#         return await get_image_service(file_id)
#     except Exception as e:
#         logger.error(f"Error getting image: {e}")
#         raise HTTPException(status_code=404, detail="File not found")



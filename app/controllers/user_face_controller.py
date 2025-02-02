from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from bson import ObjectId
import io
import uuid
import logging
from app.services.face_analysis_service import *
from app.services.machine_learning_service import *
from app.services.user_interface_service import *
from app.entity.face_image import *
from io import BytesIO
from app.repositories.face_image_repository import *
import csv
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)
# face_image_repo = face_image_repository()

@router.post("/upload/")
async def upload_image(file: UploadFile = File(...),
                       consent: bool = Query(..., description="유저 개인정보 동의"),
                       gender: str = Query(..., description="성별 (men/women)")
                       ):
    min_file_size = 10 * 1024
    if file.size < min_file_size:
        raise HTTPException(status_code=505, detail="Uploaded file is too small. Minimum size required: 10KB")

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

        prediction["predicted_job1_image"] = f"ai_images/{prediction['predicted_job1_translated']}.jpg"
        prediction["predicted_job2_image"] = f"ai_images/{prediction['predicted_job2_translated']}.jpg"
        prediction["predicted_job3_image"] = f"ai_images/{prediction['predicted_job3_translated']}.jpg"

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/secret/")
async def train():
    prediction_service = machine_learning_service()
    prediction_service.self_learn()


# CSV 파일에 데이터를 추가하는 함수
def add_to_csv(english_word: str, korean_word: str, filename="app/jop_100_1.csv"):
    file_exists = os.path.exists(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["ENGLISH", "KOREAN"])
        writer.writerow([english_word, korean_word])


# POST 요청을 처리하는 엔드포인트
@router.post("/add_word/")
async def add_word1(word_pair: WordPair):
    try:
        add_to_csv(word_pair.english, word_pair.korean)
        return {"message": f"{word_pair.english} and {word_pair.korean} added!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# CSV 파일 경로
CSV_FILE_PATH = "app/face_samples/face_specific_data.csv"
IMAGE_SAVE_PATH = "app/face_samples/"

# CSV 파일에 데이터 업데이트 함수
def update_csv(data: FaceData):
    file_exists = os.path.exists(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:  # 파일이 처음 생성될 경우 헤더 추가
            writer.writerow(["name", "gender", "job1", "job2", "job3"])
        writer.writerow([data.name, data.gender, data.job1, data.job2, data.job3])
@router.post("/update_face_data", operation_id="unique_update_face_csv")
async def update_face_data(
        name: str = Form(...),
        gender: str = Form(...),
        job1: str = Form(...),
        job2: Optional[str] = Form(None),
        job3: Optional[str] = Form(None),
        file: UploadFile = File(...)
):
    data = FaceData(name=name, gender=gender, job1=job1, job2=job2, job3=job3)
    update_csv(data)

    # 이미지 저장 경로 설정
    if not os.path.exists(IMAGE_SAVE_PATH):
        os.makedirs(IMAGE_SAVE_PATH)

    file_location = os.path.join(IMAGE_SAVE_PATH, f"{name}.jpg")
    with open(file_location, "wb") as image_file:
        image_file.write(await file.read())

    return {"message": "Data and image updated successfully", "image_path": file_location}

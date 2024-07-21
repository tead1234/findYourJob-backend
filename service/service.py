import face_recognition

# 두 이미지 파일의 경로
image1_path = "path_to_first_image.jpg"
image2_path = "path_to_second_image.jpg"

# 이미지 로드 및 얼굴 인코딩 생성
image1 = face_recognition.load_image_file(image1_path)
image1_encoding = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file(image2_path)
image2_encoding = face_recognition.face_encodings(image2)[0]

# 얼굴 유사도 비교
results = face_recognition.compare_faces([image1_encoding], image2_encoding)

print("Are the faces similar?", results[0])

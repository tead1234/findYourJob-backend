import os
import sys
from pathlib import Path
import json
from io import BytesIO
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from app.services.face_analysis_service import get_face_landmarks


class database_initialize_util:
    def __init__(self):
        pass

    @staticmethod
    def init_database():
        face_sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_samples')
        output_dir = Path(r'app')
        output_dir.mkdir(exist_ok=True)
        csv_file = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)), 'face_specific_data.csv'))
        face_data_df = pd.read_csv(csv_file, header=None, names=["filename", "gender", "job1", "job2", "job3"])

        all_face_data = {}

        for image_file in face_sample_dir.iterdir():
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    with open(image_file, 'rb') as f:
                        file_content = f.read()
                        file_stream = BytesIO(file_content)
                        landmarks = get_face_landmarks(file_stream)
                    
                    if landmarks:
                        # JSON 데이터 구성
                        face_data = {
                            "landmarks": landmarks
                        }

                        # 이미지 파일명에 해당하는 추가 정보 가져오기
                        filename = image_file.stem
                        additional_info = face_data_df[face_data_df['filename'] == filename]

                        if not additional_info.empty:
                            face_data["gender"] = additional_info["gender"].values[0]
                            face_data["job1"] = additional_info["job1"].values[0]
                            face_data["job2"] = additional_info["job2"].values[0]
                            face_data["job3"] = additional_info["job3"].values[0]

                        # 이미지를 파일명으로 데이터를 저장
                        all_face_data[filename] = face_data
                    else:
                        print(f"Warning: No landmarks detected in {image_file}")
                        continue  # 다음 이미지로 진행
                
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
                    continue
        output_file = output_dir / "init_landmark_data.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(all_face_data, f, indent=4)
            print(f"Successfully saved landmark data to {output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")



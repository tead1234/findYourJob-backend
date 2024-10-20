from ..config import *
from ..entity.face_image import face
import json


class face_image_repository:
    def __init__(self):
        self.client, self.db = get_async_gridfs()
        self.collection = self.db.face

    async def get_all_face_images(self):
        cursor = self.collection.find({})
        face_images = []
        async for document in cursor:
            face_image = face(
                id=str(document["_id"]),
                gender=document["gender"],
                landmarks=document["landmarks"],
                job1=document["job1"],
                job2=document["job2"],
                job3=document["job3"]
            )
            face_images.append(face_image)
        return face_images

    async def save_face_image(self, face_image: face) -> str:
        print("save_face_image flow")
        face_image_insert = face_image.dict()
        result = await self.collection.insert_one(face_image_insert)
        return str(result.inserted_id)

    async def save_face_data_from_json(self, json_path):
        with open(json_path, 'r') as file:
            file_content = file.read()

        json_data = json.loads(file_content)
        inserted_ids = []
        for value in json_data.items():
            document = {
                "landmarks": value[1]["landmarks"],
                "gender": value[1]["gender"],
                "job1": value[1]["job1"],
                "job2": value[1]["job2"],
                "job3": value[1]["job3"]
            }
            result = await self.collection.insert_one(document)
            inserted_ids.append(str(result.inserted_id))
        with open(r"app\id_list", 'w') as f:
            for _id in inserted_ids:
                f.write(_id + '\n')

    async def get_total_records(self):
        return await self.collection.count_documents({})

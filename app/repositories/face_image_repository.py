import motor
from app.config import DB_URL, DB_NAME
from app.entity.face_image import face


class face_image_repository:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(DB_URL)
        self.db = self.client[DB_NAME]
        self.collection = self.db.face_images

    async def get_all_face_images(self):
        cursor = self.collection.find({})
        face_images = []
        async for document in cursor:
            face_image = face(
                id=str(document["_id"]),
                gender=document["gender"],
                landmarks=document["landmarks"],
                job_1=document["job_1"],
                job_2=document["job_2"],
                job_3=document["job_3"]
            )
            face_images.append(face_image)
        return face_images

    async def save_face_image(self, face_image) -> str:
        result = await self.collection.insert_one(face_image)
        return str(result.inserted_id)

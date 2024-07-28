import motor.motor_asyncio
from app.config import DB_URL, DB_NAME

async def get_async_gridfs():
    client = motor.motor_asyncio.AsyncIOMotorClient(DB_URL)
    db = client[DB_NAME]
    return motor.motor_asyncio.AsyncIOMotorGridFSBucket(db), db
import os
import motor.motor_asyncio

DB_URL = os.getenv("DB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mydatabase")


def get_async_gridfs():
    client = motor.motor_asyncio.AsyncIOMotorClient(DB_URL)
    db = client[DB_NAME]
    print(db, "connected")
    return motor.motor_asyncio.AsyncIOMotorGridFSBucket(db), db

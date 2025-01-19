from pydantic import BaseModel
from typing import List


class face(BaseModel):
    id: str
    gender: str
    landmarks: List[List[List[int]]]
    job1: str
    job2: str
    job3: str

class WordPair(BaseModel):
    english: str
    korean: str


class FaceData(BaseModel):
    name: str
    gender: str
    job1: str
    job2: str
    job3: str
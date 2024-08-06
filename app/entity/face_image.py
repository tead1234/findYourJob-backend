from pydantic import BaseModel, Field
from typing import List


class face(BaseModel):
    id: str
    gender: str
    landmarks: List[List[int]]
    job_1: str
    job_2: str
    job_3: str

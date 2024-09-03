from pydantic import BaseModel
from fastapi import UploadFile, File

class TumorDetectionBase(BaseModel):
    file: UploadFile = File(...)

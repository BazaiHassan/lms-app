import fastapi
from fastapi import Depends, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.db_setup import async_get_db
# from utils.ml import tumor_detection
from backend.api.utils.ml import tumor_detection

router = fastapi.APIRouter()

@router.post("/tumor-detection")
async def tumor_detection_endpoint(user_id: int,file: UploadFile = File(...), db: AsyncSession = Depends(async_get_db)):
    return await tumor_detection(db=db, user_id=user_id, file=file)

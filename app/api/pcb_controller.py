from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from app.services.pcb_service import PCBService
import logging
import os
from uuid import uuid4  

logger = logging.getLogger("PCB-Defect-Detection")

pcb_router = APIRouter()

@pcb_router.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_stream, mime_type = PCBService.predict_image(file)
        return StreamingResponse(image_stream, media_type=mime_type, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        error_text = "Image prediction failed"
        logger.error(f"{error_text}: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"message": error_text, "status": "error", "details": str(e)}, 
            status_code=500
        )

PROCESSED_VIDEO_DIR = "processed_videos"
os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True)  

@pcb_router.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    try:
        video_filename = f"{uuid4()}.mp4"
        video_path = os.path.join(PROCESSED_VIDEO_DIR, video_filename)

        output_path = await PCBService.predict_video(file, video_path)

        return StreamingResponse(
            open(output_path, "rb"), media_type="video/mp4"
        )
    except HTTPException:
        raise
    except Exception as e:
        error_text = "Video prediction failed"
        logger.error(f"{error_text}: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"message": error_text, "status": "error", "details": str(e)}, 
            status_code=500
        )

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from app.services.pcb_service import PCBService
import logging

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

@pcb_router.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    try:
        video_stream, mime_type = PCBService.predict_video(file)
        return StreamingResponse(video_stream, media_type=mime_type, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        error_text = "Video prediction failed"
        logger.error(f"{error_text}: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"message": error_text, "status": "error", "details": str(e)}, 
            status_code=500
        )

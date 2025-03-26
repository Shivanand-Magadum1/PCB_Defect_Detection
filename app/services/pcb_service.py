import logging
from app.helpers.model_helper import ModelLoader
from fastapi import HTTPException

logger = logging.getLogger("PCB-Defect-Detection")

# Load model
model_loader = ModelLoader()

class PCBService:
    """Handles PCB defect detection logic and delegates processing to the model helper."""

    @staticmethod
    def predict_image(file):
        try:
            image_stream, mime_type = model_loader.process_image(file)
            if not image_stream:
                raise HTTPException(status_code=404, detail="No output generated for image")
            return image_stream, mime_type
        except HTTPException:
            raise
        except Exception as e:
            error_text = "Error in processing image"
            logger.error(f"{error_text}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"{error_text}: {str(e)}")

    @staticmethod
    def predict_video(file):
        try:
            video_stream, mime_type = model_loader.process_video(file)
            if not video_stream:
                raise HTTPException(status_code=404, detail="No output generated for video")
            return video_stream, mime_type
        except HTTPException:
            raise
        except Exception as e:
            error_text = "Error in processing video"
            logger.error(f"{error_text}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"{error_text}: {str(e)}")

import logging
import os
import tempfile
import cv2
from fastapi import HTTPException
from app.helpers.model_helper import ModelLoader

logger = logging.getLogger("PCB-Defect-Detection")

model_loader = ModelLoader()

class PCBService:
    @staticmethod
    def predict_image(file):
        try:
            image_stream, mime_type = model_loader.process_image(file)
            if not image_stream:
                raise HTTPException(status_code=404, detail="No output generated for image")
            return image_stream, mime_type
        except Exception as e:
            logger.error(f"Error in processing image: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    @staticmethod
    async def predict_video(file, output_path):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(await file.read())
                temp_video_path = temp_video.name

            output_video = model_loader.process_video(temp_video_path, output_path)

            os.remove(temp_video_path)
            return output_video
        except Exception as e:
            logger.error(f"Failed to process video: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")

import io
import os
import cv2
import numpy as np
import tempfile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from app.config.EnvConfig import EnvConfig
import logging
from fastapi import HTTPException


logger = logging.getLogger("PCB-Defect-Detection")

class ModelLoader:
    """Loads the YOLO model and processes images and videos for PCB defect detection."""

    def __init__(self):
        self.model_path = EnvConfig.YOLO_MODEL_PATH
        self.model = YOLO(self.model_path)

    def process_image(self, file):
        """Processes an image file for defect detection."""
        try:
            contents = file.file.read()
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            results = self.model(img)
            for result in results:
                img = result.plot()

            _, buffer = cv2.imencode(".jpg", img)
            return io.BytesIO(buffer.tobytes()), "image/jpeg"

        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

    def process_video(self, file):
        """Processes a video file for defect detection and returns it as a streaming response."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(file.file.read())
                temp_video_path = temp_video.name

            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Invalid video file")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(output.name, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame)
                for result in results:
                    frame = result.plot()

                out.write(frame)

            cap.release()
            out.release()

            def video_stream():
                with open(output.name, mode="rb") as video_file:
                    while chunk := video_file.read(1024 * 1024):  # 1MB chunks
                        yield chunk
                os.remove(output.name)  # Cleanup temp file

            return video_stream(), "video/mp4"

        except Exception as e:
            logger.error(f"Failed to process video: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")

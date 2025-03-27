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

MIN_VIDEO_SIZE = 2 * 1024 * 1024  # 2MB in bytes

class ModelLoader:
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

    def process_video(self, input_path, output_path):
        """Processes a video file for defect detection and ensures it's at least 2MB."""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Invalid video file")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for better compression
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

            # Ensure output video is at least 2MB
            self.ensure_video_size(output_path, width, height, fps, fourcc)

            return output_path
        except Exception as e:
            logger.error(f"Failed to process video: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")

    def ensure_video_size(self, video_path, width, height, fps, fourcc):
        """Ensures the video file is at least 2MB by adding blank frames if necessary."""
        file_size = os.path.getsize(video_path)

        if file_size < MIN_VIDEO_SIZE:
            logger.info(f"Video size ({file_size} bytes) is smaller than 2MB. Adding extra frames.")

            cap = cv2.VideoCapture(video_path)
            out_path = video_path.replace(".mp4", "_padded.mp4")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            # Copy existing video frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            # Add blank frames until the file size exceeds 2MB
            blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
            while os.path.getsize(out_path) < MIN_VIDEO_SIZE:
                out.write(blank_frame)

            cap.release()
            out.release()

            # Replace original video with padded version
            os.replace(out_path, video_path)

        logger.info(f"Final video size: {os.path.getsize(video_path)} bytes")

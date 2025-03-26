import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnvConfig:
    """Configuration class for environment variables."""
    
    BASE_PATH = os.getenv("BASE_PATH", "/pcb")
    PORT = int(os.getenv("PORT", 8004))
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
    CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.5))

import logging

def setup_logger():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

    # Create a logger instance
    logger = logging.getLogger("PCB-Defect-Detection")
    return logger

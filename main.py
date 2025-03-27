import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.pcb_controller import pcb_router
from app.config.EnvConfig import EnvConfig
from app.config.logging_config import setup_logger

setup_logger()
logger = logging.getLogger("PCB-Defect-Detection")

app = FastAPI(title="PCB Defect Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_prefix = f"{EnvConfig.BASE_PATH}/api/v1"
app.include_router(pcb_router, prefix=api_prefix, tags=["PCB Defect Detection"])

@app.get("/")
def root():
    logger.info("PCB Defect Detection API is running.")
    return {"message": "PCB Defect Detection FastAPI is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=EnvConfig.PORT)

# PCB Defect Detection API 

This FastAPI-based backend detects defects in PCB images and videos using YOLO11 by deep learning.

##  Getting Started

### 1. Create virtual environment

python -m venv env

### 2. activate venv using
venv\Scripts\activate

### 3. Install required libraries
pip install -r requirement.txt

### 4. Start the server
uvicorn main:app --reload

### 5. Access application in Browser
localhost:8005
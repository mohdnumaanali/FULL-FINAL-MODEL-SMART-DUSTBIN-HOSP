from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="Pragathi Smart Biomedical Waste API")

# Load trained YOLO model
model = YOLO("best (2).pt")

# STRICT MAPPING: Exact dataset folder names mapped to bin colors
BIN_MAPPING = {
    "1-infusion blue": "BLUE",
    "2-tubes blue": "BLUE",
    "3-hand sanitizer brown": "BLACK",
    "4-mercury thermometer brown": "BLACK",
    "5-cardboard green": "GREEN",
    "6-wrappers-images green": "GREEN",
    "7-gloves red": "RED",
    "8-syringe red": "RED",
    "9-bandage yellow": "YELLOW",
    "10-blister_strip yellow": "YELLOW"
}

# Require a high confidence score before accepting a classification
CONFIDENCE_THRESHOLD = 0.82

@app.get("/")
def home():
    return {"status": "Online", "system": "Biomedical Waste Classifier"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image payload."}

    # Run inference
    results = model(img, verbose=False)[0]
    top1_index = results.probs.top1
    detected_class = results.names[top1_index].lower().strip()
    confidence = float(results.probs.top1conf.item())

    # Check if the detected class is explicitly in our medical dataset AND meets confidence
    if confidence >= CONFIDENCE_THRESHOLD and detected_class in BIN_MAPPING:
        target_bin = BIN_MAPPING[detected_class]
        is_medical = True
        display_item = detected_class
    else:
        # Reject background, face, or unlisted random items
        target_bin = "NO MEDICAL WASTE DETECTED"
        is_medical = False
        display_item = "No valid medical item"

    return {
        "is_medical_waste": is_medical,
        "detected_item": display_item,
        "confidence_percentage": round(confidence * 100, 2),
        "target_bin": target_bin
    }
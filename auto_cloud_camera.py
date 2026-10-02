import cv2
import requests
import time
import threading

API_URL = "http://127.0.0.1:8000/predict"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access laptop webcam.")
    exit()

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

# Global variables to handle background API threading without freezing the video
latest_crop = None
is_predicting = False

prediction_data = {
    "bin": "SEARCHING...",
    "item": "Move waste object in view",
    "conf": 0.0
}

BIN_COLORS_BGR = {
    "YELLOW": (0, 255, 255),
    "RED": (0, 0, 255),
    "BLUE": (255, 105, 180),
    "GREEN": (0, 255, 0),
    "BLACK": (128, 128, 128),
    "GENERAL / UNKNOWN": (200, 200, 200)
}

def send_request_async(cropped_img):
    """Background worker function to send cropped frame to FastAPI server without lagging webcam FPS."""
    global is_predicting, prediction_data
    try:
        _, encoded = cv2.imencode('.jpg', cropped_img)
        res = requests.post(
            API_URL, 
            files={'file': ('crop.jpg', encoded.tobytes(), 'image/jpeg')}, 
            timeout=2
        )
        if res.status_code == 200:
            data = res.json()
            conf = data.get("confidence_percentage", 0.0)
            
            if conf >= 50.0:
                prediction_data["bin"] = data.get("target_bin", "UNKNOWN")
                prediction_data["item"] = data.get("detected_item", "unknown")
                prediction_data["conf"] = conf
            else:
                prediction_data["bin"] = "GENERAL / UNKNOWN"
                prediction_data["item"] = "Uncertain Object"
                prediction_data["conf"] = conf
    except Exception:
        pass
    finally:
        is_predicting = False

print("Continuous Motion Tracking Active! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 1. Real-time Frame Differencing (Motion Detection)
    frame_diff = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:  # Sensitivity threshold for motion tracking
            if area > max_area:
                max_area = area
                largest_contour = contour

    # 2. Continuous Box Tracking & Prediction Dispatch
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Padding around tracked motion region
        pad = 10
        h_img, w_img, _ = frame.shape
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

        cropped_obj = frame[y1:y2, x1:x2]

        # Trigger async API prediction whenever the server worker is free
        if cropped_obj.size > 0 and not is_predicting:
            is_predicting = True
            threading.Thread(target=send_request_async, args=(cropped_obj.copy(),), daemon=True).start()

        # 3. Draw Continuous Tracking Box on Screen
        bin_name = prediction_data["bin"]
        box_color = BIN_COLORS_BGR.get(bin_name, (255, 255, 255))

        # Box following object
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

        # Floating Box Label
        label = f"{bin_name} | {prediction_data['item']} ({prediction_data['conf']}%)"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        
        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + label_w + 10, max(20, y1)), box_color, -1)
        cv2.putText(frame, label, (x1 + 5, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    prev_gray = gray.copy()

    # Fixed Top HUD
    cv2.rectangle(frame, (10, 10), (630, 45), (0, 0, 0), -1)
    status = f"TRACKING BIN: [{prediction_data['bin']}] | ITEM: {prediction_data['item']}"
    cv2.putText(frame, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("Continuous Motion Tracking - Smart Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
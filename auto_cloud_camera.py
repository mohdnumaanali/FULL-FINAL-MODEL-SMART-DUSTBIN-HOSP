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

is_predicting = False

prediction_data = {
    "bin": "NO MEDICAL WASTE DETECTED",
    "item": "Waiting for item...",
    "conf": 0.0
}

BIN_COLORS_BGR = {
    "YELLOW": (0, 255, 255),
    "RED": (0, 0, 255),
    "BLUE": (255, 105, 180),
    "GREEN": (0, 255, 0),
    "BLACK": (128, 128, 128),
    "NO MEDICAL WASTE DETECTED": (0, 0, 255) # Bright Red for non-medical / unknown
}

def send_request_async(cropped_img):
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
            prediction_data["bin"] = data.get("target_bin", "NO MEDICAL WASTE DETECTED")
            prediction_data["item"] = data.get("detected_item", "No valid medical item")
            prediction_data["conf"] = data.get("confidence_percentage", 0.0)
    except Exception:
        pass
    finally:
        is_predicting = False

print("Strict Medical Waste Detector Active!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 1. Motion Tracking
    frame_diff = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3500:
            if area > max_area:
                max_area = area
                largest_contour = contour

    # 2. Process Detected Motion
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)

        pad = 10
        h_img, w_img, _ = frame.shape
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

        cropped_obj = frame[y1:y2, x1:x2]

        if cropped_obj.size > 0 and not is_predicting:
            is_predicting = True
            threading.Thread(target=send_request_async, args=(cropped_obj.copy(),), daemon=True).start()

        # Draw box and floating label
        bin_name = prediction_data["bin"]
        box_color = BIN_COLORS_BGR.get(bin_name, (0, 0, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

        if bin_name == "NO MEDICAL WASTE DETECTED":
            label = "NO MEDICAL WASTE DETECTED"
        else:
            label = f"BIN: {bin_name} | {prediction_data['item']} ({prediction_data['conf']}%)"

        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + label_w + 10, max(20, y1)), box_color, -1)
        cv2.putText(frame, label, (x1 + 5, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    else:
        # Reset display when no motion occurs
        prediction_data["bin"] = "NO MEDICAL WASTE DETECTED"
        prediction_data["item"] = "Waiting for item..."

    prev_gray = gray.copy()

    # Top Status HUD
    cv2.rectangle(frame, (10, 10), (630, 45), (0, 0, 0), -1)
    status = f"STATUS: {prediction_data['bin']}"
    cv2.putText(frame, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("Medical Waste Filtered Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
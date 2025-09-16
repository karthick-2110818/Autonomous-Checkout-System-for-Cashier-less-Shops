import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time
import requests
from hx711 import HX711

# Initialize HX711 Load Cell
GPIO.setmode(GPIO.BCM)
hx = HX711(dout_pin=20, pd_sck_pin=21)
hx.set_scale_ratio(420.641)
hx.zero()

# Define Freshness Classes
classes = [
    "Fresh-Apple", "Fresh-Mango", "Fresh-Banana", "FreshBellpepper", "FreshCarrot",
    "FreshCucumber", "FreshOrange", "FreshPotato", "FreshStrawberry", "FreshTomato",
    "Rotten-Apple", "RottenBanana", "RottenBellpepper", "RottenCarrot", "RottenCucumber",
    "RottenMango", "RottenOrange", "RottenPotato", "RottenStrawberry", "RottenTomato"
]

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8 Model
model = YOLO("best.onnx")

# Define pricing per gram for fresh items
pricing = {
    "Fresh-Apple": 0.05, "Fresh-Mango": 0.06, "Fresh-Banana": 0.03,
    "FreshBellpepper": 0.04, "FreshCarrot": 0.02, "FreshCucumber": 0.03,
    "FreshOrange": 0.05, "FreshPotato": 0.01, "FreshStrawberry": 0.10,
    "FreshTomato": 0.04
}

# Backend URL
url = "https://backend-for-fyp.onrender.com"

# Initialize Frame Counter
frame_count = 0

# Main Loop
while True:
    frame = picam2.capture_array()
    
    # Skip frames to reduce Raspberry Pi load (process every 3rd frame)
    frame_count += 1
    if frame_count % 3 != 0:
        continue  # Skip processing and move to the next frame

    frame = cv2.flip(frame, -1)
    # Flip frame for correct orientation

    # Run YOLOv8 Inference
    results = model.predict(frame, imgsz=256)

    # Process Detection Results
    for result in results:
        boxes = result.boxes.xyxy.numpy() if result.boxes.xyxy is not None else []
        class_ids = result.boxes.cls.numpy() if result.boxes.cls is not None else []
        confidences = result.boxes.conf.numpy() if result.boxes.conf is not None else []

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            class_name = classes[int(class_id)]
            x1, y1, x2, y2 = map(int, box)

            # Get weight from load cell
            weight = hx.get_weight_mean(20)
            price_per_gram = pricing.get(class_name, 0)
            total_price = round(weight * price_per_gram, 2)

            # Send confirmed detection to the backend
            data = {
                "name": class_name,
                "weight": round(weight, 2),
                "price": total_price,
                "freshness": "Fresh" if "fresh" in class_name.lower() else "Rotten"
            }
            headers = {"Content-Type": "application/json"}
            try:
                response = requests.post(f"{url}/product", json=data, headers=headers)
                if response.status_code == 200:
                    print(f"Sent: {class_name}, Weight: {weight:.2f}g, Price: Rs.{total_price:.2f}")
                else:
                    print(f"Failed to send {class_name}. Status: {response.status_code}")
            except Exception as e:
                print(f"Error sending to backend: {e}")

            # Draw Bounding Box on Frame
            color = (0, 255, 0) if "Fresh" in class_name else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Display Frame
    cv2.imshow("Autonomous Checkout", frame)

    # Exit Condition
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean Up
picam2.stop()
GPIO.cleanup()
cv2.destroyAllWindows()


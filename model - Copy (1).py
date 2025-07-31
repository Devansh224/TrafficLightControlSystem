import cv2
import numpy as np
import time
# import serial  
from yolov5 import YOLOv5  # Using proper YOLOv5 package
import pathlib 
pathlib.PosixPath = pathlib.WindowsPath

# Optional: Arduino connection
# arduino = serial.Serial("COM8", 9600, timeout=1) 

# Classes mapping
CLASS_MAP = {
    0: 'car',
    1: 'ambulance',
    3: 'bus',
    4: 'fire_truck'
}

# Load your trained model
model = YOLOv5('best.pt', device='cpu')

# Previous positions (optional future use)
previous_positions = {}

# Skip speed estimation for miniatures
def estimate_speed(prev_pos, curr_pos):
    return 0

# Timer logic with emergency vehicle priority
def determine_timer(vehicle_count, emergency_detected):
    if emergency_detected:
        return 15  # Force timer to 15 seconds for emergency
    unit_time = 8  # seconds per vehicle
    timer = (vehicle_count / 2) * unit_time
    return min(30, max(5, round(timer)))

# Vehicle detection and control function
def detect_vehicles(image):
    global previous_positions
    results = model.predict(image)

    lane1_count, lane2_count = 0, 0
    emergency_detected = False

    for box in results.pred[0]:
        x1, y1, x2, y2, conf, cls_id = map(float, box[:6])
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        vehicle_type = CLASS_MAP.get(int(cls_id), 'unknown')
        vehicle_id = hash((x1, y1, x2, y2))  # Simple tracking

        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        curr_position = (x_center, y_center)

        # Future: Speed estimation
        prev_position = previous_positions.get(vehicle_id)
        speed = estimate_speed(prev_position, curr_position)
        previous_positions[vehicle_id] = curr_position

        # Check for emergency vehicles
        if vehicle_type in ['ambulance', 'fire_truck']:
            emergency_detected = True

        # Count vehicles per lane
        if x_center < image.shape[1] // 2:
            lane1_count += 1
        else:
            lane2_count += 1

        # Draw bounding box and label
        color = (0, 255, 0) if vehicle_type not in ['ambulance', 'fire_truck'] else (0, 255, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{vehicle_type}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Timer calculations
    timer1 = determine_timer(lane1_count, emergency_detected)
    timer2 = determine_timer(lane2_count, emergency_detected)

    # Optional: send to Arduino
    # arduino.write(f"{timer1},{timer2}\n".encode())

    # Display overlay info
    cv2.putText(image, f"Lane 1: {lane1_count} vehicles", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, f"Lane 2: {lane2_count} vehicles", (image.shape[1] // 2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if emergency_detected:
        print("🚨 Emergency vehicle detected! Giving priority to all lanes.")

    print(f"🚦 Lane 1 GREEN for {timer1}s | Lane 2 RED")
    # time.sleep(timer1)

    print(f"🚦 Lane 2 GREEN for {timer2}s | Lane 1 RED")
    # time.sleep(timer2)

    return image

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        detected_frame = detect_vehicles(frame)
        cv2.imshow('Traffic Detection', detected_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    # arduino.close()

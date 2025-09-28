import cv2
import numpy as np
import cv2
import math
from ultralytics import YOLO


# Load YOLOv8n pretrained model
model = YOLO("home/codetech/Desktop/CodeAlpha_Object-detection-and-tracking /yolo/yolo11n.pt")


#YUsing webcam
cap = cv2.VideoCapture(1)

# Initialize
count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    center_points_cur_frame = []

    # Run YOLO detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes  # YOLO detection boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box
            conf = float(box.conf[0])               # confidence
            cls = int(box.cls[0])                   # class id
            class_name = model.names[cls]           # class name (e.g. "person")

            # Center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            center_points_cur_frame.append((cx, cy))

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Tracking logic
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            if not object_exists:
                tracking_objects.pop(object_id)

        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    # Draw tracking IDs
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {object_id}", (pt[0], pt[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

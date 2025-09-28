import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import math

class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection & Tracking Interface")
        self.root.geometry("800x600")

        self.model = None
        self.file_path = None

        # Tracking variables
        self.count = 0
        self.center_points_prev_frame = []
        self.tracking_objects = {}
        self.track_id = 0

        # Buttons
        self.load_model_btn = tk.Button(root, text="Load Detection Model", command=self.load_model)
        self.load_model_btn.pack(pady=10)

        self.load_file_btn = tk.Button(root, text="Load Image/Video", command=self.load_file, state=tk.DISABLED)
        self.load_file_btn.pack(pady=10)

        self.run_btn = tk.Button(root, text="Run Prediction", command=self.run_prediction, state=tk.DISABLED)
        self.run_btn.pack(pady=10)

        # Canvas for display
        self.canvas = tk.Label(root)
        self.canvas.pack()

    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if model_path:
            self.model = YOLO(model_path)
            messagebox.showinfo("Model Loaded", f"Loaded model: {model_path}")
            self.load_file_btn.config(state=tk.NORMAL)

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Media", "*.jpg *.jpeg *.png *.mp4")])
        if self.file_path:
            self.run_btn.config(state=tk.NORMAL)

    def run_prediction(self):
        if not self.model or not self.file_path:
            messagebox.showerror("Error", "Load model and file first")
            return

        # Image case
        if self.file_path.endswith((".jpg", ".jpeg", ".png")):
            results = self.model(self.file_path)
            img = results[0].plot()
            self.display_image(img)

        # Video case with tracking
        elif self.file_path.endswith(".mp4"):
            self.cap = cv2.VideoCapture(self.file_path)
            self.process_video()

    def process_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        self.count += 1
        center_points_cur_frame = []

        results = self.model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]

                # Center
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                center_points_cur_frame.append((cx, cy))

                # Draw box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Tracking logic
        if self.count <= 2:
            for pt in center_points_cur_frame:
                for pt2 in self.center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    if distance < 20:
                        self.tracking_objects[self.track_id] = pt
                        self.track_id += 1
        else:
            tracking_objects_copy = self.tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    if distance < 20:
                        self.tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                        continue

                if not object_exists:
                    self.tracking_objects.pop(object_id)

            for pt in center_points_cur_frame:
                self.tracking_objects[self.track_id] = pt
                self.track_id += 1

        # Draw tracking IDs
        for object_id, pt in self.tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {object_id}", (pt[0], pt[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show frame in Tkinter
        frame = cv2.resize(frame, (1000, 800))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk

        self.center_points_prev_frame = center_points_cur_frame.copy()
        self.root.after(10, self.process_video)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()

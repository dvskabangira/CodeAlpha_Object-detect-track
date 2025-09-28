import cv2
import numpy as np



'''
model = YOLO("yolo8n.pt")

#Train the model
model_results = model.train(
                data = "coco8.yaml",
                epochs = 100,
                imgsize = 640,
                device  = 0
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/home/codetech/Desktop/CodeAlpha_Object-detection-and-tracking /yolo/image.jpg")
results[0].show()

# Export the model
path = model.export()
'''




from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/home/codetech/Desktop/CodeAlpha_Object-detection-and-tracking /yolo/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model





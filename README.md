# CodeAlpha_Objection detection & tracking
In this project, we use YOLO (yolo11n & yolo8n) for object detection. And for object tracking, we save the center point of each object, trace the previous position of the objects
and predict what the immediate next will be. At first, we initialize an array to keep track of the previous points, then we calculate the distance between the points to ensure they all belong to the same object. The closer the points are, the greater the probability that we are tracking the same object.
If a new object is identified, the list of points is updated



***Results***
<p align="center">
    <img src="https://github.com/dvskabangira/CodeAlpha_Object-detect-track/blob/main/prediction_output.png", width="540">

**REAL-TIME FACE RECOGNIZE**

Using YOLO, DeepSort, Transfer learning of VGG16 to make a real-time face recognize.

# YOLO
Using pre-trained model yolov8n-face.pt to detect face from camera
Link download model: https://github.com/akanametov/yolov8-face

# DeepSort
Using DeepSort to track the face in the camera

# Transfer learning of VGG16
Using VGG16 to make a transfer learning model which help to recognize the face.
Keep the extract features layers from image of the model.
Replace the classifier layers to fit the requirements.

# Results 

<p align="center">
  <video src="video/video.mp4" width="500px"></video>
</p>

<video src="video/video.mp4" width="320" height="240" controls></video>

![non working video](images/video.mp4)













# FaceDetectRecog_ZEDCam
This is a face detection and recognition model using a ZED Camera. 
The ZED Camera is a stereo camera which helps to get depth perception to calculate distance to objects in its view.
Along with image processing and Machine Learning, the ZED API was used to access camera and get images.

The main code consists of a dataset creating function where the dataset of 100 images is stored in a sub-folder specific to a person
haarcascades is used for face detection and recognition is doen with a help of Local Binary Pattern Histogram Method.
The prediction is decided based on the confidence (distance) of the prediction from the image dataset
The distance calculation is done by point cloud method from the ZED API

My contribution is the code for detection and recognition of faces and the function to append new faces to the dataset.

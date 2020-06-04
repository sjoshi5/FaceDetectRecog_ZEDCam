# Detection and Recognition using ZED Camera
import cv2, sys, os, math
import numpy as np
import pyzed.sl as sl

camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
init_params.coordinate_units = sl.UNIT.UNIT_METER  # Use meter units (for depth measurements)
cam = sl.Camera()
status = cam.open(init_params)
mat = sl.Mat()
runtime = sl.RuntimeParameters()
def dataset():
    haar_file = 'haarcascade_frontalface_default.xml'

    # Faces are stored in this folder
    datasets = 'datasets'

    # Every person is saved as a separate folder
    sub_data = input('Enter Name')

    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)

    # Defining fixed size for image
    (width, height) = (130, 100)

    face_cascade = cv2.CascadeClassifier(haar_file)
    # webcam = cv2.VideoCapture(0)

    # Capturing 100 images for dataset
    count = 1
    while count < 100:
        (_, jkl) = webcam.read()
        im = jkl[0:720, 0:514]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('% s/% s.png' % (path, count), face_resize)
        count += 1


haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
o = 0
count = 0
# Creating Fisher Recognizer
print('Recognizing Face Please Be in sufficient Lights...')

# Creating a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (130, 100)

# Creating a Numpy array using the lists of images and names
(images, lables) = [np.array(lis) for lis in [images, lables]]

# Training model using LBPH
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)

# Use Fisher Recognizer on video feed
face_cascade = cv2.CascadeClassifier(haar_file)
init = sl.InitParameters()
webcam = sl.Camera()
runtime_parameters = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD  # Use STANDARD sensing mode

# Capture 50 images and depth, then stop
i = 0
image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()


while True:
    err = cam.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.VIEW_RIGHT)
        im = mat.get_data()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.VIEW_LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.MEASURE_DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA)

            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            x = round(image.get_width() / 2)
            y = round(image.get_height() / 2)
            err, point_cloud_value = point_cloud.get_value(x, y)

            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])

            if not np.isnan(distance) and not np.isinf(distance):
                distance = round(distance)
                print("Distance to Camera at ({0}, {1}): {2} mm\n".format(x, y, distance))
                # Increment the loop
                i = i + 1
            else:
                print("Can't estimate distance at this position, move the camera\n")
            sys.stdout.flush()

        if prediction[1] < 90:

            cv2.putText(im, '% s - %.0f' %
                        (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(im, 'not recognized',
                        (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        count = count + 1
        if count == 50:
            if o == 0:
                print('Add person? (Y/N)')
                resp = input('Enter Response')
                if resp == 'y':
                    dataset()
                o = 1

    cv2.imshow('OpenCV', im)

    # key = cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Capture
webcam.release()
cv2.destroyAllWindows()

# Close the camera
zed.close()
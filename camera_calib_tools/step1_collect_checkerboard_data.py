import numpy as np
import cv2
import glob, os, time
import video_channels
import select_camera_to_calibrate

# helper script to collect imagery of checkerboard 
# when you are ready to start saving images press 's'. It will then save an image ervy second until you press 'q'

config = video_channels.channel(select_camera_to_calibrate.Camera)

try:
    os.mkdir(config.Folder)
except:
    print("folder already exists. Proceeding to add to folder: "+config.Folder)

images = glob.glob(config.Folder+'*.jpg')

numExistingIms = len(images)

cap = cv2.VideoCapture(config.CapturePath)

cv2.namedWindow('cameraview',cv2.WINDOW_NORMAL)
counter = 0
collectionStarted = False
while(True):
    ret, frame = cap.read()
    if counter == 0:
        assert config.imSize[0] == frame.shape[0] and config.imSize[1] == frame.shape[1], "The frame is of a different size than what is specified in the configuration file (video_channels.py)"
    thisTime = time.time()
    cv2.imshow('cameraview',cv2.flip(frame, 1))
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('s'):
        collectionStarted = True
    if collectionStarted and counter % 25 == 0:
        numExistingIms += 1
        cv2.imwrite(config.Folder+'cb_im'+str(numExistingIms)+'.jpg',frame)
    counter += 1


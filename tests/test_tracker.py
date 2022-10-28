import cv2
import numpy as np
import camera
import human_tracker

cap = cv2.VideoCapture('../sampledata/towncentre/TownCentreXVID.avi')
cv2.namedWindow('cameraview',cv2.WINDOW_AUTOSIZE | cv2. WINDOW_KEEPRATIO)
r, img0 = cap.read()

# set up rectangles to move along and test that they are tracked correctly
imSize = img0.shape[0:2]

numRects = 25
initPos = 1080 * np.ones((numRects,2),np.int)
initPos[:,0] = np.linspace(100,imSize[1]-100,numRects)

numRects = initPos.shape[0]
VelocityMinPerMiles = 15
speedMSec = 1600 / (VelocityMinPerMiles * 60)
Speeds = -np.ones(initPos.shape)
Speeds[:,0] = 0.15 * np.sin(np.arange(numRects)*500.0)
Speeds = 30 * speedMSec * np.divide(Speeds,np.expand_dims(np.linalg.norm(Speeds,axis=1),axis=1))

Widths = np.round(60  + 5 * np.sin(np.arange(numRects)*1400.0)).astype(np.int)
Heights = np.round(150  + 5 * np.sin(np.arange(numRects)*1400.0)).astype(np.int)

myTracker = human_tracker.tracker(img0.shape, '../sampledata/towncentre/TownCentre-calibration.ci.txt', 50, False)
Pos = initPos.copy()

Time = 0.0
delta_t = 1.0 / 25.0
interv = 5
Delta_t = delta_t * interv
Counter = 0
while (True):
    img = img0.copy()
    if (Counter % 5 == 0):
        # move the objects
        Pos += (np.round(Speeds * Delta_t + 5 * (1 - np.random.rand(numRects,2)))).astype(np.int)
        toReset = np.where(np.logical_or(np.logical_or(Pos[:,1] < 300, Pos[:,0] < 25),Pos[:,0] + Widths > imSize[1]-25))[0]
        Pos[toReset,:] = initPos[toReset,:] 
        boxesRaw = []
        for i in range(numRects):
            boxesRaw.append((Pos[i,1]-Heights[i],Pos[i,0],min(1080,Pos[i,1]),Pos[i,0]+Widths[i]))
        
        boxes,annotation = myTracker.report_detections(boxesRaw,0.95 * np.ones(numRects,np.float),np.ones(numRects,np.int),img, numRects,Time)
    else:
        boxes, annotation = myTracker.estimate_new_positions(Time, img)
    # Visualization of the results of tracking.
    for i in range(len(boxes)):
        # Class 1 represents human
        box = boxes[i]
        cv2.rectangle(img ,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(255,0,0),2)
        # write something too 
        cv2.putText(img,annotation[i],(int(box[1]),int(box[0])), cv2.FONT_ITALIC,  1,(255,0,0),2)

    cv2.imshow('cameraview', img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break    
        
    Time += delta_t
    Counter += 1


cv2.destroyWindow('cameraview')
cap.release()

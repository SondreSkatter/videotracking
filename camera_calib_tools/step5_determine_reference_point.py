import numpy as np
import cv2
import video_channels
import camera
import calibrate_camera
import select_camera_to_calibrate

myConfig = video_channels.channel(select_camera_to_calibrate.Camera)
# make sure step1-4 are completed0
# this script will be looking for a elongated white line (tape). This will be the y axis, 
# There should also be a dot over the line (like an i); this indicates positive y direction
# and y = 0 will be at the center of the dot


# keystrokes for zoom '+', unzoom '-', pan rigth 'r', pan left 'l', 'u', 'd'
# to select white line: press 's'
# to accept selection: press 'a'
try:
    myCamera = camera.camera(myConfig.imSize[0:2], myConfig.calScale, myConfig.calib_path+"preXYref.txt", myConfig.useGeometricCenterAsRotAxis)
except:
    raise Exception('step 1-4 needs to be completed!')

cv2.namedWindow('cameraview',cv2.WINDOW_NORMAL)
zoomState = 1.0
zoomStep = 0.35
leftrightstate = 0
updownstate = 0
panStep = 50

xPix, yPix = np.meshgrid(np.linspace(0, myCamera.n1-1, myCamera.n1), np.linspace(0, myCamera.n2-1, myCamera.n2))    
imShape = xPix.shape
xMap, yMap = myCamera.getGroundPosition(xPix.flatten(),yPix.flatten())
xMap = np.reshape(xMap,imShape)
yMap = np.reshape(yMap,imShape)

cap = cv2.VideoCapture(myConfig.CapturePath)
drawCaption = True
keepLooping = True
while(keepLooping):
    ret, frame = cap.read()
    # draw things in the frame now...

    centerX = int(myCamera.n1/2 + leftrightstate)
    centerV = int(myCamera.n2/2 + updownstate)
    Left = max(0,int(centerX - myCamera.n1/2/zoomState))
    Right = min(myCamera.n1,int(centerX + myCamera.n1/2/zoomState))
    Top = max(0,int(centerV - myCamera.n2/2/zoomState))
    Bottom = min(myCamera.n2,int(centerV + myCamera.n2/2/zoomState))
    #oh, and write instructions too
    if drawCaption:
        cv2.putText(frame,'Point to the reference marker (white line). Select reference by pressing s.  Zoom and pan +,-,r, l, u, d.',(Left+50,Top+50), cv2.FONT_ITALIC,  1,(0,0,255),3)

    cv2.imshow('cameraview',frame[Top:Bottom,Left:Right])
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('c'):
        numExistingIms += 1
        cv2.imwrite(myStream.Folder+'cb_im'+str(numExistingIms)+'.jpg',frame)
    elif key & 0xFF == ord('+'):
        zoomState += zoomStep
    elif key & 0xFF == ord('-'):
        if zoomState > 1.0:
            zoomState -= zoomStep
    elif key & 0xFF == ord('u'):
        updownstate -= panStep
    elif key & 0xFF == ord('d'):
        updownstate += panStep
    elif key & 0xFF == ord('l'):
        leftrightstate -= panStep
    elif key & 0xFF == ord('r'):
        leftrightstate += panStep
    elif key & 0xFF == ord('s'):
        # will search for the marker, only inside the zoomed view
        im = frame[Top:Bottom,Left:Right]
        imSize = im.shape

        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        # Define range of white color in HSV
        if 0:
            Mask = cv2.inRange(hsv, (0, 0, 0), (190, 30,255))
            Mask = cv2.bitwise_and(Mask,cv2.inRange(hsv, (0, 0, 220), (190, 255,255)))
        else:
            if myConfig.camName == "axis-pole":
                Mask = cv2.inRange(hsv, (0, 0, 150), (190, 100,255))
                Mask = cv2.inRange(hsv, (0, 0, 140), (190, 110,255))
                Mask = cv2.inRange(hsv, (0, 0, 170), (190, 60,255))
            elif myConfig. camName == "axis-wall":
                Mask = cv2.inRange(hsv, (0, 0, 170), (190, 60,255))
        
        whiteness = cv2.cvtColor(Mask,cv2.COLOR_GRAY2BGR) #Mask.astype(np.float)
        windowName = 'test'
        cv2.namedWindow('test',cv2.WINDOW_NORMAL)
        cv2.putText(whiteness,'Press y if youre happy with the thresholding of white from backgroudn and n if not.',(50,50), cv2.FONT_ITALIC,  1,(0,0,255),3)
        cv2.imshow('test',whiteness)
        keepLooping = True
        while keepLooping:
            key = cv2.waitKey(1)
            if key & 0xFF == ord('y'):
                break
            elif key & 0xFF == ord('n'):
                keepLooping = False

        if not keepLooping:
            break
        #kernel = np.ones((3,3), np.uint8)
        ret, labels = cv2.connectedComponents(Mask)
        (y2,x2) = np.where(Mask.astype(bool))
        bb,ibb,nbb = np.unique(labels[[y2],[x2]],return_inverse=True,return_counts=True)
        numObj = bb.shape[0]
        
        if numObj >= 2:
            
            # good we got the dotted i, presumably
            Means = np.zeros((2,numObj),np.float)
            eigVals = np.zeros((2,numObj),np.float)
            eigVecs = np.zeros((2,2,numObj),np.float)
            for i in range(numObj):
                inds = np.where(ibb == i)[0]

                if inds.size < 4:
                    continue
                x = x2[inds] + Left
                y = y2[inds] + Top

                Pos2D = np.vstack((xMap[[y],[x]] , yMap[[y],[x]]))
                Cov = np.cov(Pos2D)
                
                eigVals[:,i], eigVecs[:,:,i] = np.linalg.eig(Cov); 
                Means[:,i] = np.mean(Pos2D,axis=1)
            
            lineInd = np.argmax(np.sum(eigVals,axis=0))                        
            Axis = eigVecs[:,np.argmax(eigVals[:,lineInd]),lineInd]
            # Now we want the rotation matrix to be such that it turns Axis to the [0,1]
            rotMat = np.array([[Axis[1],-Axis[0]],[Axis[0],Axis[1]]])

            transCenters = np.matmul(rotMat,Means)
            # find the dot now
            dotInd = -1

            for i in range(numObj):
                if i != lineInd and np.abs(transCenters[0,i] - transCenters[0,lineInd]) < 0.030:
                    dotInd = i

            if dotInd == -1:
                raise Exception('Could not locate the dot near the line...')
            # now make sure that the dot has a higher y value than the line...

            #dotInd = int(1 - lineInd)
            #dotPos = np.matmul(rotMat,Means[:,dotInd])
            #linePos = np.matmul(rotMat,Means[:,lineInd])
            dotPos = transCenters[:,dotInd]
            linePos = transCenters[:,lineInd]

            whiteness[:] = 0
            inds = np.where(ibb == lineInd)[0]
            whiteness[y2[inds],x2[inds],0] = 255
            inds = np.where(ibb == dotInd)[0]
            whiteness[y2[inds],x2[inds],1] = 255


            cv2.putText(whiteness,'Press y if youre happy with the selection of the i and n if not.',(50,50), cv2.FONT_ITALIC,  1,(0,0,255),3)
            cv2.imshow('test',whiteness)
            keepLooping = True
            while keepLooping:
                key = cv2.waitKey(1)
                if key & 0xFF == ord('y'):
                    break
                elif key & 0xFF == ord('n'):
                    keepLooping = False

            if not keepLooping:
                break


            if dotPos[1] < linePos[1]:
                # turn another 180 degrees
                rotMat = -rotMat
            # now we need to add this rotation to the quaternion, then determine the Translation in x and y from the center of linePos (Means[:,lineInd]
            theta = -np.arctan2(rotMat[0,1],rotMat[0,0])
            quatZ = np.array([np.cos(theta/2), 0.0, 0.0, np.sin(theta/2)])
            quatComb = camera.quaternion_mult(np.expand_dims(quatZ,axis=1), np.expand_dims(myCamera.quat,axis=1)).flatten()
            TransXY = -np.matmul(rotMat,Means[:,dotInd])
            Trans = np.append(TransXY,myCamera.trans[2])
            camPars = np.array([myCamera.fx,myCamera.fy,myCamera.cx,myCamera.cy,myCamera.k1,myCamera.k2,myCamera.p1,myCamera.p2,myCamera.k3])
            camParsClassic = np.array([myCamera.k1class,myCamera.k2class,myCamera.p1class,myCamera.p2class,myCamera.k3class])

            calibrate_camera.saveCamCalFile(myConfig.calib_path, camPars, camParsClassic, Trans, quatComb)
            break
        d = 1
        key = cv2.waitKey(5000)    


cap.release()
cv2.destroyAllWindows()
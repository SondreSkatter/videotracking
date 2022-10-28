import numpy as np
import cv2
import glob, os, shutil
import video_channels
import pickle
import calibrate_camera
import select_camera_to_calibrate

# needs there to be images collected in step 1
# will write as output camLensParams.npz in the camera folder

config = video_channels.channel(select_camera_to_calibrate.Camera)
optimizeNow = True
show_undistorted = False

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

cameraMatrixguess = np.zeros((3,3), np.float)
cameraMatrixguess[2,2] = 1.0
cameraMatrixguess[0,2] =  config.imSize[1] / 2.0
cameraMatrixguess[1,2] =  config.imSize[0] / 2.0

cv2.namedWindow('cameraview',cv2.WINDOW_NORMAL)
images = glob.glob(config.Folder+'*.jpg')
badImageFolder = config.Folder + 'badChessboardImgs/'
goodImageFolder = config.Folder + 'goodChessboardImgs/'
annotatedGoodImageFolder = config.Folder + 'goodChessboardImgs/annot/'
try: 
    os.mkdir(badImageFolder)
    os.mkdir(goodImageFolder)
    os.mkdir(annotatedGoodImageFolder)
except:
    pass
badImages = glob.glob(badImageFolder+'*.jpg')
numBadImgs = len(badImages)
goodImages = glob.glob(goodImageFolder+'*.jpg')
numGoodImgs = len(goodImages)
if numGoodImgs > 0:
    f = open(goodImageFolder + "cached_results", "rb")
    objpoints = pickle.load(f)
    imgpoints = pickle.load(f)
    max_r = pickle.load(f)
    f.close() 
else:
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    max_r = np.zeros(0, np.int)

numNewSamples = 0

Pattern = (9,7)
for i,fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, Pattern, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(fname,Pattern)
        numNewSamples += 1
        objpoints.append(objp)

        r = np.sqrt(np.square(corners[:,0,1] - config.imSize[0] / 2.0) + np.square(corners[:,0,0] - config.imSize[1] / 2.0)) / (config.imSize[1] / 2.0)
        max_r = np.append(max_r, np.max(r))
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, Pattern, corners2,ret)
        cv2.imshow('cameraview',img)
        cv2.waitKey(1)           
        numGoodImgs += 1
        shutil.move(fname, goodImageFolder+'cb_im'+str(numGoodImgs)+'.jpg')
        cv2.imwrite(annotatedGoodImageFolder+'cb_im'+str(numGoodImgs)+'.jpg',img)
    else:
        # couldn't find all the corners. Will just move this image to another folder
        numBadImgs += 1
        shutil.move(fname, badImageFolder+'cb_im'+str(numBadImgs)+'.jpg')
if numNewSamples > 0:
    f = open(goodImageFolder + "cached_results", "wb")
    pickle.dump(objpoints,f)
    pickle.dump(imgpoints,f)
    pickle.dump(max_r,f)
    f.close() 
    

cv2.destroyAllWindows()

# We need to stabilize the solutions, so we'll take advantage of the measurements done in step 2
try:
    drawPixPos = np.load(config.Folder+'drawPos.npz')
    measDists = np.load(config.Folder+'measuredDistances.npz')
    # we have three points in space, we know the position in image and we can compute the true angle between the points 
    pixDist = np.zeros(3)
    deltaTan = np.zeros(3)
    pixDist[0] = np.linalg.norm(drawPixPos['drawPos'][1,:]-drawPixPos['drawPos'][0,:])
    pixDist[1] = np.linalg.norm(drawPixPos['drawPos'][2,:]-drawPixPos['drawPos'][1,:])
    pixDist[2] = np.linalg.norm(drawPixPos['drawPos'][1,:]-drawPixPos['drawPos'][3,:])


    DirVecs = np.array(((-measDists['LeftToMiddle'],np.sqrt(np.square(measDists['LeftDiag'])-np.square(measDists['LeftToMiddle'])),-measDists['camHeight']),\
        (0.0,measDists['midDiag'],-measDists['camHeight']),\
        (-measDists['RightToMiddle'],np.sqrt(np.square(measDists['RightDiag'])-np.square(measDists['RightToMiddle'])),-measDists['camHeight']),
                        (0.0,measDists['midDiag']+measDists['midToFar'],-measDists['camHeight'])))

    DirVecs = DirVecs / np.expand_dims(np.linalg.norm(DirVecs,axis=1),axis=1)
    
    pixPoses = drawPixPos['drawPos'][0:-1,:]
    deltaTan[0] = np.tan(np.arcsin(np.linalg.norm(np.cross(DirVecs[0,:],DirVecs[1,:]))))
    deltaTan[1] = np.tan(np.arcsin(np.linalg.norm(np.cross(DirVecs[2,:],DirVecs[1,:]))))
    deltaTan[2] = np.tan(np.arcsin(np.linalg.norm(np.cross(DirVecs[3,:],DirVecs[1,:]))))

    yFraction = abs((drawPixPos['drawPos'][1,1] - drawPixPos['drawPos'][3,1]) / (drawPixPos['drawPos'][1,1] - config.imSize[0]/2))
    deltaTanY = deltaTan[2] / yFraction
    deltaTanYSmall = deltaTanY * (1.0 - yFraction)

    TanXY = np.array(( (-deltaTan[0], deltaTanY), (0,deltaTanY) , (deltaTan[1],deltaTanY), (0,deltaTanYSmall) ))    
    cameraMatrixguess[0,0] = pixDist[-1] / deltaTan[-1]
    cameraMatrixguess[1,1] = cameraMatrixguess[0,0] 

except:
    assert True, 'Be sure to complete steps 2 & 2b then try again. '


if optimizeNow:
    print("optimizing the distortion parameters.... will take a while.")
    if 1:
        maxTanx = np.tan(0.5 * config.lens_angle_coverage)
        mtx, dist = calibrate_camera.calibrate_camera(objpoints, imgpoints, cameraMatrixguess, np.array([-0.5*0, 2.0*0, 0.0, 0.0, 0.0]), max_r,maxTanx,config.imSize, TanXY, pixPoses)
    else:    
        retVal, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (config.imSize[0],config.imSize[1]), cameraMatrixguess, np.array([-0.3, 0.5, 0.0, 0.0, 0.0]))

    np.savez(config.Folder+'camLensParams',fx=mtx[0,0],fy=mtx[1,1],cx=mtx[0,2],cy=mtx[1,2],k1=dist[0],k2=dist[1],k3=dist[4],p1=dist[2],p2=dist[3],n1=config.imSize[1],n2=config.imSize[0])
    # OpenCV helped us find the distortion parameters, but we actually need to move in the opposite direction: from pixels to 3D positions    
    distInv = calibrate_camera.invertDistortion(config.imSize[0:2], mtx, dist)
    np.savez(config.Folder+'camLensParamsInv',fx=mtx[0,0],fy=mtx[1,1],cx=mtx[0,2],cy=mtx[1,2],k1=distInv[0],k2=distInv[1],k3=distInv[4],p1=distInv[2],p2=distInv[3],n1=config.imSize[1],n2=config.imSize[0])
    print("done...")
else:
    camParams = np.load(config.Folder+'camLensParams.npz')
    camPars = np.array([camParams['fx'],camParams['fy'],camParams['cx'],camParams['cy'],camParams['k1'],camParams['k2'],camParams['p1'],camParams['p2'],camParams['k3']])
    mtx, dist = calibrate_camera.getMtrxFromParams(camPars)

if show_undistorted:
    # try out udistorting
    calibrate_camera.show_calibrated_imgs(glob.glob(goodImageFolder+'*.jpg'), mtx, dist)




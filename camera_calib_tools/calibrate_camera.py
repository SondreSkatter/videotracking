import cv2, numpy as np

def getParamsFromMtrx(mtrx, dist):
    return np.array([mtrx[0,0],mtrx[1,1],mtrx[0,2],mtrx[1,2],dist[0],dist[1],dist[2],dist[3],dist[4]])

def getMtrxFromParams(Params):

    mtrx = np.zeros((3,3), np.float)
    mtrx[0,0] = Params[0]
    mtrx[1,1] = Params[1]
    mtrx[2,2] = 1.0
    mtrx[0,2] = Params[2]
    mtrx[1,2] = Params[3]
    dist = Params[4:].copy()
    return mtrx, dist

def lineError(objpoints,imgpoints,Inds,camParams):
    mtrx, dist = getMtrxFromParams(camParams)
    Error = 0.0
    overFlowScale = 1E-8
    DOF = 0
    for i in Inds:
        ret, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], mtrx, dist)
        #ret, rvec, tvec, inliers = cv2.solvePnPRansac(objpoints[i], imgpoints[i], mtrx, np.zeros(5,np.float))
        imgpts, jac = cv2.projectPoints(objpoints[i], rvec, tvec, mtrx, dist)
        theseErrors = np.clip(np.square(imgpoints[i]-imgpts),0,1E6)
        Error += np.sum(theseErrors * overFlowScale) # thisError[i]
        DOF += objpoints[i].shape[0]
    return (Error / DOF) / overFlowScale

def optimizeLineError(objpoints, imgpoints, Inds, paramsToTune, MeanParams, CovParams, stepLengths, numTrials, lockFxtoFy, maxTanx, imSize, applyConstraint):
     
    assert not lockFxtoFy or (lockFxtoFy and (np.any(paramsToTune == 1) and np.any(paramsToTune == 0))), "when using lockFxtoFy include both fx and fy"
    
    Means = MeanParams[paramsToTune]
    Covs = CovParams[np.ix_(paramsToTune,paramsToTune)]
   
    numKeepers = 150
    
    numSections = int(round(numTrials/(numKeepers*2)))

    numTrials = int(np.floor(numTrials / numSections) * numSections)
    Error = np.zeros(numTrials, np.float)
    nInSec = int(numTrials / numSections)
    camParams = np.expand_dims(MeanParams,axis=1) * np.ones(numTrials, np.float)
    for s in range(numSections):
        #print("running batch ",s," of ",numSections," batches.")
        inds = nInSec*s +  np.arange(nInSec)

        if (s > 0):
            # can we do something more? Using Error as weight perhaps??
            prevInds = np.arange(inds[0])
            prevInds = prevInds[np.logical_not(np.isnan(Error[prevInds]))]
            weights = 1.0 / (Error[prevInds] + 0.01)
            
            sortInd = np.argsort(weights)

            weights[sortInd[0:(weights.size-numKeepers)]] = 0.0
            weights /= np.sum(weights)
        
            Params = camParams[np.ix_(paramsToTune,prevInds)]
            if len(Params.shape) == 1:
                Params = np.expand_dims(Params,axis=0)

            Means = np.sum(Params * weights, axis=1)
            Res = Params - np.expand_dims(Means,axis=1)
            Covs = np.matmul(Res * weights, Res.transpose())
    
        if len(Covs.shape) == 1:
            Covs = np.expand_dims(Covs,axis=1)

        if applyConstraint:
            overshootFactor = 1.2
            while True:
                newPars = np.expand_dims(MeanParams,axis=1) * np.ones(int(nInSec*overshootFactor), np.float)
                newPars[paramsToTune,:] = np.random.multivariate_normal(Means, Covs, int(nInSec*overshootFactor)).transpose()
                # Compute compatibility with constraint
                maxPixX = newPars[0,:] * maxTanx * (1.0 + newPars[4,:] * maxTanx ** 2 + newPars[5,:] * maxTanx ** 4 + newPars[8,:] * maxTanx ** 6)
                initErrors = np.square(maxPixX - 0.5 * imSize[1]) 

                # also, we need the derivative to remain positive in the corners...
                maxTanCorner = 1.025 * np.sqrt(maxTanx**2 + (3/4*maxTanx)**2)
                testVals = np.expand_dims(np.linspace(0.0,maxTanCorner,20),axis=1)
                deriv = 1 + 3 * newPars[4,:] * testVals ** 2 + 5*newPars[5,:] * testVals**4 + 7 * newPars[8,:] * testVals ** 6 
                initErrors += 1E8 * (np.any(deriv < 0, axis=0))

                goodInds = np.where(initErrors < (imSize[1]/15)**2)[0]
                numGoodPoints = goodInds.size
                if numGoodPoints < nInSec:
                    overshootFactor *= 2
                else:
                    camParams[:,inds] = newPars[:,goodInds[0:nInSec]]
                    if lockFxtoFy and np.any(paramsToTune == 0):
                        camParams[1,inds] = camParams[0,inds]
                    break

        else:
            camParams[np.ix_(paramsToTune,inds)]  = np.random.multivariate_normal(Means, Covs, nInSec).transpose()        
        for i in inds:
            Error[i] = lineError(objpoints,imgpoints,Inds,camParams[:,i])
        print("Stage ",s+1," of ", numSections," completed. Average error for new points: ",np.mean(Error[inds]),", fx sigma: ",np.sqrt(Covs[0,0]))


    minInd = np.argmin(Error)
    minErr = Error[minInd]
    bestParams = camParams[:,minInd]
    MeanParams[paramsToTune] = Means

    CovParams[np.ix_(paramsToTune,paramsToTune)] = Covs
    # compute error for mean also
    ErrorAtMean = lineError(objpoints,imgpoints,Inds,MeanParams)
    return np.sqrt(minErr), np.sqrt(ErrorAtMean), MeanParams, bestParams, CovParams

def calibrate_camera(objPts, imgpoints, cameraMatrixguess, distGuess, max_r,maxTanx,imSize, TanXY, pixPoses):

    keepRatioFixed = True
    paramLabels = ['fx','fy','cx','cy','k1','k2','p1','p2','k3','k4']
    paramValues0 = getParamsFromMtrx(cameraMatrixguess, distGuess)
    stepLengths = np.array([0.25, 0.25, 0.5, 0.5, 0.01, 0.01, 0.005, 0.005, 0.01, 0.01])
    Range = np.array([paramValues0[0]*0.05, paramValues0[0]*0.05, paramValues0[3]*0.05, paramValues0[3]*0.05, 0.75, 2, 0.5, 0.5, 1, 1]) 

    #stepLengths = np.array([0.25, 0.25, 0.5, 0.5, 0.01, 0.01, 0.005, 0.005, 0.01, 0.01])
    Range = np.array([paramValues0[0]*0.05, paramValues0[0]*0.5, paramValues0[3]*0.05, paramValues0[3]*0.05, 0.75, 2, 0.01, 0.01, 1, 1]) 

    CovParams = np.diag(np.square(Range))

    # To start out, we will impose some constraints in the covar matrix based on the measured positions
    # first: fx/fy & k1 
    Pars = np.array([0,4])
    Means = paramValues0[Pars]
    Covs = np.diag(np.square(Range[Pars]))
    numInitials = 5000
    numKeepers = numInitials//2# 200
    numRounds = 3

    r2Vals = np.expand_dims(np.expand_dims(np.square(np.linalg.norm(TanXY,axis=1)),axis=1),axis=2) 
    TanXY = np.expand_dims(TanXY,axis=-1)
    
    for i in range(numRounds):
        camParamsInit  = np.random.multivariate_normal(Means, Covs, numInitials).transpose()
        camParamsInit[:,0] = Means # just for testing
        # Take 3 points given by TanXY and compute pixel position
        PredPos = np.expand_dims(np.expand_dims(paramValues0[np.array((2,3))],axis=0),axis=2) + \
           TanXY * camParamsInit[0,:]*(1+camParamsInit[1,:]*r2Vals)
        Errors = np.linalg.norm(PredPos -np.expand_dims( pixPoses, axis=-1), axis=1).squeeze()
        initErrors = np.mean(Errors,axis=0)

        weights = 1.0 / (initErrors + 15**2)
            
        sortInd = np.argsort(weights)

        weights[sortInd[0:(weights.size-numKeepers)]] = 0.0
        weights /= np.sum(weights)
        
        Params = camParamsInit
        if len(Params.shape) == 1:
            Params = np.expand_dims(Params,axis=0)

        Means = np.sum(Params * weights, axis=1)
        Res = Params - np.expand_dims(Means,axis=1)
        Covs = np.matmul(Res * weights, Res.transpose())

    paramValues0[Pars] = Means
    paramValues0[1] = Means[0]
    CovParams[np.ix_(Pars,Pars)] = Covs
    Pars[0] = 1
    CovParams[np.ix_(Pars,Pars)] = Covs
    CovParams[0,1] = CovParams[0,0] 
    CovParams[1,0] = CovParams[0,0] 

    newCamParams = paramValues0

    allSamples = np.arange(max_r.size)

    # just get a benchmark error for the new dataset
    ErrorRef = np.sqrt(lineError(objPts,imgpoints,allSamples,newCamParams)) 

    numTrials = 6000

    Pars = np.array([0, 1, 2, 3, 4, 5, 8])

    minErr,newErr,newCamParams,bestCamParams,CovParams = optimizeLineError(objPts, imgpoints, allSamples, Pars, newCamParams, CovParams, stepLengths[Pars], numTrials, keepRatioFixed,maxTanx,imSize,1)

    if 0:
        keepRatioFixed = False
        numTrials = 3000
        CovParams[0,1] = 0.5 * CovParams[0,0]
        CovParams[1,0] = CovParams[0,1]  
        print("Decoupling fx and fy")
        minErr,newErr,newCamParams,bestCamParams,CovParams = optimizeLineError(objPts, imgpoints, allSamples, Pars, newCamParams, CovParams, stepLengths[Pars], numTrials, keepRatioFixed,maxTanx,imSize,1)

    if 0: # the tangential portion
        print("Including tangential components")
        numTrials = 4000
        keepRatioFixed = False
        Pars = np.arange(9)
        minErr,newErr,newCamParams,bestCamParams,CovParams = optimizeLineError(objPts, imgpoints, allSamples, Pars, newCamParams, CovParams, stepLengths[Pars], numTrials, keepRatioFixed, maxTanx,imSize,0)

    return getMtrxFromParams(newCamParams)

def invertDistortion(imSize, mtx, dist):
    # create 3D angles in a grid....

    # Need to probe things out to know how big angles we need in the grid...
    # Start out with a huge angle, then trim back to ones insside the pixel size..
    LensAngle = 2.5
    N = 3000
    N2 = int(N/2)
    testAngle = np.linspace(-LensAngle*0.5,LensAngle*0.5,N)
    obPts = np.array([[testAngle],[np.zeros(testAngle.size,np.float)],[np.ones(testAngle.size,np.float)]]).squeeze().transpose()
    imgpts, jac = cv2.projectPoints(obPts, np.zeros(3,np.float), np.zeros(3,np.float), mtx, dist)

    minXind = np.max(np.where(imgpts[0:N2,0,0] < 0.0)[0])
    maxXind = N2 + np.min(np.where(imgpts[N2:,0,0] > imSize[1]-1)[0])
    obPts = np.array([[np.zeros(testAngle.size,np.float)],[testAngle],[np.ones(testAngle.size,np.float)]]).squeeze().transpose()
    imgpts, jac = cv2.projectPoints(obPts, np.zeros(3,np.float), np.zeros(3,np.float), mtx, dist)

    minYind = np.max(np.where(imgpts[0:N2,0,1] < 0.0)[0])
    maxYind = N2 + np.min(np.where(imgpts[N2:,0,1] > imSize[0]-1)[0])

    xv, yv = np.meshgrid(np.linspace(testAngle[minXind],testAngle[maxXind],1000), np.linspace(testAngle[minYind],testAngle[maxYind],700), sparse=False, indexing='ij')

    xv = xv.ravel()
    yv = yv.ravel()

    # apply model now
    obPts = np.array([[xv],[yv],[np.ones(xv.size,np.float)]]).squeeze().transpose()

    imgpts, jac = cv2.projectPoints(obPts, np.zeros(3,np.float), np.zeros(3,np.float), mtx, dist)

    goodInds = np.where(np.logical_and(np.logical_and(imgpts[:,0,0] >= 0.0, imgpts[:,0,0] <= imSize[1]-1),np.logical_and(imgpts[:,0,1] >= 0.0, imgpts[:,0,1] <= imSize[0]-1)))[0]

    nPts = goodInds.size


    includeTangential = np.any(np.abs(dist[2:4]) > 1E-5)

    if includeTangential:
        A = np.zeros((2*nPts,5), np.float)
    else:
        A = np.zeros((2*nPts,3), np.float)

    x1 = (imgpts[goodInds,0,0] - mtx[0,2]) / mtx[0,0]
    y1 = (imgpts[goodInds,0,1] - mtx[1,2]) / mtx[1,1]
    r2 = x1 * x1 + y1 * y1

    rMax = np.sqrt(np.max(r2))

    bVec = np.expand_dims(np.append(xv[goodInds]-x1, yv[goodInds]-y1),axis=1)

    # k1
    A[0:nPts,0] = x1 * r2
    A[nPts:,0] = y1 * r2
    # k2
    A[0:nPts,1] = x1 * r2 * r2
    A[nPts:,1] = y1 * r2 * r2
    # k3
    A[0:nPts,2] = x1 * r2 * r2 * r2
    A[nPts:,2] = y1 * r2 * r2 * r2
    if includeTangential:
        # p1
        A[0:nPts,3] = 2 * x1 * x1 * y1 
        A[nPts:,3] = y1 * (r2 + 2 * y1 * x1)
        # p2
        A[0:nPts,4] = x1 * (r2 + 2 * y1 * x1) 
        A[nPts:,4] = 2 * x1 * y1 * y1 

    # now, just solve it...
    Sol = np.linalg.lstsq(A,bVec)
    if includeTangential:
        Sol2 = Sol[0][np.array((0,1,3,4,2)),0] 
    else:
        Sol2 = np.array((Sol[0][0,0],Sol[0][1,0],0.0,0.0,Sol[0][2,0]))
    return Sol2

def show_calibrated_imgs(impaths, mtx, dist):
    cv2.namedWindow('cameraview',cv2.WINDOW_NORMAL)
    for i, fname in enumerate(impaths):
        img = cv2.undistort(cv2.imread(fname), mtx, dist, None)
        cv2.imshow('cameraview',img)
        cv2.waitKey(1000)     
    cv2.destroyWindow('cameraview')

def readCamCalFile(calFile):
    f = open(calFile, 'r')
    x = f.readlines()
    f.close()
    camParams = np.zeros(9, np.float)
    trans = np.zeros(3, np.float)
    quat = np.zeros(4, np.float)

    camParams[0] = float(x[0].split(' = ')[1])
    camParams[1] = float(x[1].split(' = ')[1])
    camParams[2] = float(x[2].split(' = ')[1])
    camParams[3] = float(x[3].split(' = ')[1])

    trans[0] = float(x[5].split(' = ')[1]) 
    trans[1] = float(x[6].split(' = ')[1]) 
    trans[2] = float(x[7].split(' = ')[1]) 

    quat[1] = float(x[8].split(' = ')[1])
    quat[2] = float(x[9].split(' = ')[1])
    quat[3] = float(x[10].split(' = ')[1])
    quat[0]  = float(x[11].split(' = ')[1])

    camParams[4] = float(x[12].split(' = ')[1])
    camParams[5] = float(x[13].split(' = ')[1])
    camParams[8] = float(x[14].split(' = ')[1])
    camParams[6] = float(x[15].split(' = ')[1])
    camParams[7] = float(x[16].split(' = ')[1])  
    return camParams, trans, quat

def saveCamCalFile(filepath, camParams, camParamsClass, trans, quat):
    with open(filepath, "w") as text_file:
        print("FocalLengthX = {}".format(camParams[0]), file=text_file)
        print("FocalLengthY = {}".format(camParams[1]), file=text_file)
        print("PrincipalPointX = {}".format(camParams[2]), file=text_file)
        print("PrincipalPointY = {}".format(camParams[3]), file=text_file)
        print("Skew = 0.00000000000000000000", file=text_file)
        print("TranslationX = {}".format(trans[0]), file=text_file)
        print("TranslationY = {}".format(trans[1]), file=text_file)
        print("TranslationZ = {}".format(trans[2]), file=text_file)
        print("RotationX = {}".format(quat[1]), file=text_file)
        print("RotationY = {}".format(quat[2]), file=text_file)
        print("RotationZ = {}".format(quat[3]), file=text_file)
        print("RotationW = {}".format(quat[0]), file=text_file)
        print("DistortionK1 = {}".format(camParams[4]), file=text_file)
        print("DistortionK2 = {}".format(camParams[5]), file=text_file)
        print("DistortionK3 = {}".format(camParams[8]), file=text_file)
        print("DistortionP1 = {}".format(camParams[6]), file=text_file)
        print("DistortionP2 = {}".format(camParams[7]), file=text_file)
        print("DistortionK1Classic = {}".format(camParamsClass[0]), file=text_file)
        print("DistortionK2Classic = {}".format(camParamsClass[1]), file=text_file)
        print("DistortionK3Classic = {}".format(camParamsClass[4]), file=text_file)
        print("DistortionP1Classic = {}".format(camParamsClass[2]), file=text_file)
        print("DistortionP2Classic = {}".format(camParamsClass[3]), file=text_file)
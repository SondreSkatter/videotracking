import os, cv2, time, camera, numpy as np
import video_channels
import human_detector_pose_est, posenet_helper, pose_analyzer
import scene_manager
import reconstruct_planar_elev_map
import pickle
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import norm
import tracks, chidistdf
from scipy.ndimage import gaussian_filter

def selectBestPlane(Pos,Cov,Planes,zRange,returnProbs=False):
    M = Planes.shape[0] + 1
    N = Pos.shape[0]
    Probs = np.zeros((N,M))
    Probs[:,0] = 1.0 / (zRange[1]-zRange[0])  # kind of the background class
    for j in range(M-1):
        for i in range(N):
            Norm = Planes[j,0:3] 
            Var = np.matmul(Norm,np.matmul(Cov[i,0:3,0:3],np.expand_dims(Norm,axis=1)))
            Probs[i,j+1] = np.exp(-0.5 * np.square(np.sum(Norm * Pos[i,0:3]) - Planes[j,3]) / Var) / np.sqrt(2*np.pi*Var)

    if 0:
        # plot it...
        Label = np.argmax(Probs,axis=1)
        un,ia = np.unique(Label,return_inverse=True)
        for i in range(un.size):
            inds = np.where(ia==i)[0]
            plt.plot(Pos[inds,0],Pos[inds,1], marker=i,linestyle='None')
        plt.show()

    if returnProbs:
        return Probs
    else:
        return np.argmax(Probs,axis=1)

def findBestCut(Pos,Labels,inpInds):
    # first: determine the centroid
    inds = np.where(Labels[inpInds]==1)
    Mean = np.mean(Pos[inpInds[inds],0:2],axis=0)
    Cov = np.cov(Pos[inpInds[inds],0:2].transpose())
    AngleRes = np.array((36,6,1))
    Len = np.trace(Cov) ** 0.5
    LenRes = np.array((0.2*Len, 0.5, 0.05))
    Pos0 = Pos[inpInds,0:2] - Mean
    Score0 = np.sum(np.square(Labels[inpInds]))
    for it in range(AngleRes.size):
        if it == 0:
            an =  np.arange(AngleRes[it],360+AngleRes[it],AngleRes[it])
            Dist = np.arange(LenRes[it],5*Len+LenRes[it],LenRes[it])
        else:
            an = bestAn + np.arange(-6.0,7.0) * AngleRes[it]
            Dist = bestDist + np.arange(-LenRes[it-1],LenRes[it-1]+LenRes[it],LenRes[it])
        
        a = np.cos(an/180*np.pi)
        b = np.sin(an/180*np.pi)
        c = Dist
        Score = np.zeros((an.size,Dist.size),np.float)
        for i in range(a.size):
            for j in range(c.size):
                Class = 2*(a[i] * Pos0[:,0] + b[i] * Pos0[:,1] < c[j]).astype(np.float) - 1
                Score[i,j] = np.sum(Class * Labels[inpInds])

        if 0:
            d = 1
            #figure
            #imagesc(Dist,an,Score/Score0)
            #xlabel('Distance')
            #ylabel('Angle')

        bestI,bestJ = np.unravel_index(np.argmax(Score, axis=None), Score.shape)        
        bestAn = an[bestI]
        bestDist = Dist[bestJ]
    maxScore = Score[bestI,bestJ] / Score0
    # now just need to translate back using the Mean Pos
    Line = np.array((np.cos(bestAn/180*np.pi),np.sin(bestAn/180*np.pi),bestDist))
    Line[2] += Mean[0] * Line[0] + Mean[1] * Line[1]

    remainingInds = inpInds[np.where(a[bestI] * Pos0[:,0] + b[bestI] * Pos0[:,1] < c[bestJ])[0]]
    return Line, maxScore, remainingInds

def drawClassLines(Pos,Class):
    # We use the convention that Class = 0 is the background and we won't try to delineate it
    numClasses = np.max(Class)
    Lines = [None] * numClasses
    Scores = [None] * numClasses

    maxCuts = 10
    for i in range(numClasses):
        Labels = 2 * (Class == (i+1)).astype(np.float) - 1
        Labels[Class == 0] = -0.5 # unassigned, less bad to get wrong...
        inds = np.arange(Class.size)
        lastScore = 0.0
        Lines[i] = np.zeros((0,3),np.float)
        Scores[i] = np.zeros(0,np.float)
        for j in range(maxCuts):
            # brute force, try all possible lines (?)        
            Line, Score, inds = findBestCut(Pos,Labels,inds)        
            if Score > lastScore + 0.02:
                Lines[i] = np.vstack((Lines[i],Line))
                Scores[i] = np.append(Scores[i],Score)
                lastScore = Score
            else:
                break
  
    return Lines, Scores
    
def findInterfaces(Pos,locLines):
    Overlaps0 = findOverlaps(Pos,locLines)
    numClasses = len(locLines)
    interfaceLine = np.zeros((numClasses,numClasses,3))
    interfaceInds = -np.ones((numClasses,numClasses),np.int)
    interfaceCount = np.zeros((numClasses,numClasses),np.int)
    interfaceWeight = np.zeros((numClasses,numClasses)) 
    perturbAmount = 1.5
    for i in range(numClasses):
        for j in range(Lines[i].shape[0]):
            locLines[i][j,2] += perturbAmount
            OverlapDiff = findOverlaps(Pos,locLines) - Overlaps0
            locLines[i][j,2] -= perturbAmount
            ind2,ind1 = np.unravel_index(np.argmax(OverlapDiff,axis=None),(numClasses,numClasses))
            if OverlapDiff[ind1,ind2] > 0.0001:
                Wght = OverlapDiff[ind1,ind2]
                interfaceWeight[ind1,ind2] += Wght
                if interfaceCount[ind1,ind2] > 0:
                    # already a line here, so we need to mix the two
                    # check if it's reversed
                    if np.sum(locLines[i][j,0:2] * interfaceLine[ind1,ind2,0:2]) < 0.0:
                        newLine = -locLines[i][j,:].copy()
                    else:
                        newLine = locLines[i][j,:].copy()
                    interfaceLine[ind1,ind2,:] += Wght * newLine 
                else:
                    interfaceLine[ind1,ind2,:] = Wght * locLines[i][j,:]
                if i == ind1:
                    interfaceInds[ind1,ind2] = j
                elif i == ind2:
                    interfaceInds[ind2,ind1] = j
                interfaceCount[ind1,ind2] += 1
            

    # then normalize
    inds = np.argwhere(interfaceCount>0)
    for i in range(inds.shape[0]):
        ind1 = inds[i,0]
        ind2 = inds[i,1]
        interfaceLine[ind1,ind2,:] /= interfaceWeight[ind1,ind2]
        interfaceLine[ind1,ind2,0:2] /= np.linalg.norm(interfaceLine[ind1,ind2,0:2])

    return interfaceLine,interfaceCount, interfaceInds

def findOverlaps(Pos, classLines):
    Clsses = Classify(Pos,classLines)
    Overlaps = np.matmul(Clsses.transpose(),Clsses)
    Overlaps = Overlaps / np.sqrt(np.expand_dims(np.diag(Overlaps),axis=1) * np.diag(Overlaps))
    return Overlaps

def Classify(Pos,Lines):
    numClasses = len(Lines)
    Class = np.ones((Pos.shape[0],numClasses),np.int)
    for i in range(numClasses):
        for j in range(Lines[i].shape[0]):
            Class[:,i] *= (Pos[:,0]*Lines[i][j,0] + Pos[:,1]*Lines[i][j,1] < Lines[i][j,2])
    return Class

def computePlaneDeviance(Planes,Points):
    # compute the distance to each plane and pick the closest one
    Dist = np.zeros((Points.shape[0],Planes.shape[0]))
    for i in range(Planes.shape[0]):
        Dist[:,i] = np.matmul(Points[:,0:3] , np.expand_dims(Planes[i,0:3], axis=1))[:,0] - Planes[i,3]

    bestPlane = np.argmin(np.abs(Dist),axis=1)
    # we want the z distance though...
    Error = np.mean(np.square(Dist[np.arange(Points.shape[0]),bestPlane])) 
    return bestPlane, Error

useCachedResults = False
analyzeIt = True

if 1:
    myChannels = [ video_channels.channel('wildtrack1','cache'), video_channels.channel('wildtrack2','cache'), video_channels.channel('wildtrack3','cache'), video_channels.channel('wildtrack4','cache'), video_channels.channel('wildtrack5','cache'),  video_channels.channel('wildtrack6','cache'),  video_channels.channel('wildtrack7','cache')]
else:
    recordingName = 'Dec17-2019-1321'  # on stage and with boxes
    myChannels = [ video_channels.channel('warrior-pole',recordingName), video_channels.channel('warrior-wall',recordingName) ]

parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(myChannels[0].Folder)), os.pardir))
numChannels = len(myChannels)

if useCachedResults:
    if 0:
        f = open(parent_path+"/elevationData.npz", "rb")
        Pos = pickle.load(f)
        Cov = pickle.load(f)
        Scores = pickle.load(f)
        origHeight = pickle.load(f)
        frameInd = pickle.load(f)
        channelInd = pickle.load(f)
        unitVectors = pickle.load(f)
        keyPointScores = pickle.load(f)
        N = Scores.size
        f.close()
    else:        
        f = open(parent_path+"/elevationDataReconciled.npz", "rb")
        Pos = pickle.load(f)
        Cov = pickle.load(f)
        numObs = pickle.load(f)
        N = numObs.size
        f.close()   
else:
    
    poseSolver = pose_analyzer.poseSolver([])

    cameras = []
    Paths = []
    timeToReadIms = 0.0
    timeToInitCameras = 0.0
    for c in range(numChannels):
        t1 = time.time()
        cap = cv2.VideoCapture(myChannels[c].CapturePath)
        r, img = cap.read()
        cap.release()
        t2 = time.time()
        cameras.append(camera.camera(img.shape, myChannels[c].calScale, myChannels[c].calib_path, myChannels[c].useGeometricCenterAsRotAxis) ) 
        Paths.append(os.path.dirname(myChannels[c].CapturePath)+'/cached_det_results/')
        t3 = time.time()
        timeToReadIms += t2 - t1
        timeToInitCameras += t3 - t2

    print('Retrieved cam images in {:.2f} seconds and initialized camera in {:.2f} seconds'.format(timeToReadIms,timeToInitCameras))

    numAllocItems = 1000
    allocSteps = 1000
    N = 0
    Pos = np.zeros((numAllocItems,4))
    Cov = np.zeros((numAllocItems,4,4))
    Scores = np.zeros((numAllocItems))
    origHeight = np.zeros(numAllocItems)    
    frameInd = np.zeros(numAllocItems,np.int)
    channelInd = np.zeros(numAllocItems,np.int)
    unitVectors = np.zeros((numAllocItems,3,poseSolver.goodInds.size))
    keyPointScores = np.zeros((numAllocItems,poseSolver.myPoseHelper.numParts))


    timeBegin = time.time()
    stopOnFrame = -1
    fr = 0
    wereDone = False
    while fr != stopOnFrame and not wereDone:    
        print("frame: " + str(fr))
        for c in range(numChannels):
            detResult = human_detector_pose_est.get_frame(fr, Paths[c])
            if not detResult: 
                wereDone = True
                break
            elif not detResult["detectionWasRun"]:
                continue
            BBs = poseSolver.getBBfromPose(detResult["keypoint_scores"], detResult["keypoint_coords"])  
            for i in np.where(detResult["scores"] > 0.5)[0]:              
                heightInPixels = BBs[i,2] - BBs[i,0]
                Results = poseSolver.get_person_coords_from_keypoints(detResult["keypoint_coords"][i,:,:], detResult["scores"][i], detResult["keypoint_scores"][i,:], heightInPixels, cameras[c], unknownElevation = True)
                if len(Results) == 5 and Results[0] == True:
                    Pos[N,:] = Results[1]
                    Cov[N,:,:] = Results[2]
                    origHeight[N] = Results[3]
                    unitVectors[N,:,:] = Results[4]
                    keyPointScores[N,:] = detResult["keypoint_scores"][i,:]
                    Scores[N] = detResult["scores"][i]
                    frameInd[N] = fr
                    channelInd[N] = c
                    N += 1
                if N >= numAllocItems:
                    numAllocItems += allocSteps
                    Pos.resize((numAllocItems,4))
                    Cov.resize((numAllocItems,4,4))
                    Scores.resize((numAllocItems))
                    origHeight.resize(numAllocItems)
                    frameInd.resize(numAllocItems)
                    channelInd.resize(numAllocItems)
                    unitVectors.resize((numAllocItems,3,poseSolver.goodInds.size))
                    keyPointScores.resize((numAllocItems,poseSolver.myPoseHelper.numParts))
        fr += 1
    
    print(str(N) + ' poses procesed, time per pose analyze: {:.2f} msec, per frameL {:.3f} sec'.format((time.time() - timeBegin) / N * 1000, (time.time() - timeBegin) / (fr * numChannels)))
    Pos = Pos[0:N,:]
    Cov = Cov[0:N,:,:]
    Scores = Scores[0:N]
    origHeight = origHeight[0:N]
    frameInd = frameInd[0:N]
    channelInd = channelInd[0:N]
    unitVectors = unitVectors[0:N,:,:]
    keyPointScores = keyPointScores[0:N,:]

    f = open(parent_path+"/elevationData.npz", "wb")
    pickle.dump(Pos, f)
    pickle.dump(Cov, f)
    pickle.dump(Scores, f)
    pickle.dump(origHeight, f)    
    pickle.dump(frameInd, f)  
    pickle.dump(channelInd, f)  
    pickle.dump(unitVectors, f)  
    pickle.dump(keyPointScores, f) 
    f.close()
    # save for matlab too
    sio.savemat(parent_path+"/elevationData.mat", {'Pos':Pos, 'Cov':Cov, 'Scores':Scores, 'origHeight':origHeight, 'channelInd':channelInd, 'frameInd':frameInd, 'unitVectors':unitVectors, 'keyPointScores':keyPointScores } )

    trckCfg = {  # config options
        "targetPeopleDetectFPS": 5,
        "recordData": False, # record new data set.  if set, no algorithms will run other than capturing data at the right sampling frequency
        "runLiveGPUdetection": True,
        "num_GPU_workers": 1,
        "num_orchestrators": 2,
        "cacheDetResults": True,
        "dropFramesWhenBehind": False,  # if live capture we always may have to drop frames, but this flag, if True, will force frame dropping also when running recorded data
        "usePersistedStaticObjects": False,
        "detectStaticObjects": False,
        "doFaceID": False, 
        "runTrackerInSepProcess": True,
        # display and video options
        "doRendering": True,
        "saveVideo": True,
        "showStaticObjects": False,
        "showVisitorMap": False,
        "showGT": 2, # 0: don't show, 1: show GT boxes, 2: color objects according to correct wrt GT
        "cycleDet": False, # for example, if two cameras, ping pong between them. Since GPU is the scarce compute resource
        "globalGT": [],
        "minNumSightings": 1,
        "rec_name": [],
        "liveCapture": False # will be updated based on the channel inputs
        }

    # first: reconcile poses from different channels to reduce the height uncertainty
    SceneMgr = scene_manager.scene_manager(myChannels,trckCfg,None,None,unknownHeight=True)
    #SceneMgr.minProb = np.array((0.25,0.25,0.25))
    #SceneMgr.minLogRatios = np.log(np.array((1.5,1.5)))
    SceneMgr.doOrientation = False
    numAllocItems = 1000
    allocSteps = 1000
    NN = 0
    Pos2 = np.zeros((numAllocItems,4))
    Cov2 = np.zeros((numAllocItems,4,4))
    numObs = np.zeros(numAllocItems,np.int)

    for fr in np.unique(frameInd):
        if fr % 10 == 0:
            print('Frame: '+str(fr))
        inds = np.where(fr == frameInd)[0]
        channels = np.unique(channelInd[inds])
        N = channels.size
        goodInds = [None] * N
        colorFeats = [None] * N
        colorCov = [None] * N
        hasColor = [None] * N
        posAndHghtLoc = [None] * N 
        posHeightCovLoc = [None] * N 
        poseTypeProbLoc = [None] * N 
        ScoresLoc = [None] * N 
        OrientationsLoc = [None] * N 
        unitVectorsLoc = [None] * N 
        keyPtsScrLoc = [None] * N 
        timeStamps = np.zeros(N,np.float)


        for c2, c in enumerate(channels):
            inds2 = inds[np.where(c == channelInd[inds])[0]]
            M = inds2.size
            goodInds[c2] = np.arange(M)
            colorFeats[c2] = np.zeros((M,SceneMgr.numColorFeats))
            colorCov[c2] = np.zeros((M,SceneMgr.numColorFeats,SceneMgr.numColorFeats))
            hasColor[c2] = np.zeros(M, np.bool)
            posAndHghtLoc[c2] = np.expand_dims(Pos[inds2,:],axis=-2)
            posHeightCovLoc[c2] = np.expand_dims(Cov[inds2,:,:],axis=-3)
            poseTypeProbLoc[c2] = np.ones((M,5),np.float)
            poseTypeProbLoc[c2][:,1:5] = 0.0
            ScoresLoc[c2] = Scores[inds2]
            OrientationsLoc[c2] = np.zeros((M,SceneMgr.numAngleBins))
            unitVectorsLoc[c2] = unitVectors[inds2,:,:]  
            keyPtsScrLoc[c2] = keyPointScores[inds2,:]

        SceneMgr.match_humans_in_frame_only(channels, goodInds, timeStamps, posAndHghtLoc, posHeightCovLoc, poseTypeProbLoc, ScoresLoc, colorFeats, colorCov, hasColor, OrientationsLoc, unitVectorsLoc, keyPtsScrLoc)
 
        for i in SceneMgr.myTempTracks.Inds:
            if np.sum(SceneMgr.myTempTracks.numObs[i,:]) > 0: # and SceneMgr.myTracks.posCov[i,0,2,2] < 0.85:
                Pos2[NN,:] = SceneMgr.myTempTracks.posAndHeight[i,0,:]
                Cov2[NN,:,:] = SceneMgr.myTempTracks.posCov[i,0,:,:]
                numObs[NN] = np.sum(SceneMgr.myTempTracks.numObs[i,:])
                NN += 1
                if NN >= numAllocItems:
                    numAllocItems += allocSteps
                    Pos2.resize((numAllocItems,4))
                    Cov2.resize((numAllocItems,4,4))
                    numObs.resize(numAllocItems)
            # delete all tracks to start fresh at next frame
            #SceneMgr.myTracks.removeHuman(i)
        SceneMgr.myTempTracks.clear()
    Pos = Pos2[0:NN,:]
    Cov = Cov2[0:NN,:,:]
    numObs = numObs[0:NN]
    f = open(parent_path+"/elevationDataReconciled.npz", "wb")
    pickle.dump(Pos, f)
    pickle.dump(Cov, f)
    pickle.dump(numObs, f)
    f.close()
    # save for matlab too
    sio.savemat(parent_path+"/elevationDataReconciled.mat", {'Pos':Pos, 'Cov':Cov, 'numObs':numObs } )


if analyzeIt:

    # start doing something with that data now...
    xLims = np.array([-3.0, 9.0])
    yLims = np.array([-9.0, 26.0])
    xLimsOuter = np.array([-15.0, 20.0])
    yLimsOuter = np.array([-15.0, 35.0]) 
    stepSize = 0.5
    goodInds = np.where((numObs>0) * (Cov[:,2,2] < np.square(0.25)))[0]
    N = Pos.shape[0]
    binWidth = 0.01
    bins = np.arange(np.floor(np.min(Pos[:,2] / binWidth))*binWidth,np.ceil(np.max(Pos[:,2]/binWidth))*binWidth+binWidth,binWidth)

    Histo = np.zeros(bins.size)
    HistoSharp = np.zeros(bins.size)
    sharpCov = np.square(0.01)
    for i in goodInds:
        Histo += np.exp(-0.5 * np.square(bins -  Pos[i,2]) / Cov[i,2,2]) / np.sqrt(Cov[i,2,2])
        HistoSharp += np.exp(-0.5 * np.square(bins -  Pos[i,2]) / sharpCov) / np.sqrt(sharpCov)
    
    locMax = 1 + np.where((Histo[1:-1] > Histo[0:-2]) * (Histo[1:-1] >= Histo[2:]) * (Histo[1:-1] > 0.1*np.mean(Histo)))[0]
    
    # find the more accurata maximum by looking at the sharp histogram
    for i2 in range(locMax.size):
        i = locMax[i2]
        while i > 0 and i < HistoSharp.size-1:
            if HistoSharp[i+1] > HistoSharp[i]:
                i += 1
            elif HistoSharp[i-1] > HistoSharp[i]:
                i -= 1
            else:
                break
        locMax[i2] = i
        
    if 1:
        plt.plot(bins,Histo)
        plt.plot(bins,HistoSharp)
        plt.show()

    # starting with a set of horizontal planes
    horPlanes = bins[locMax]
    horPlanes[np.abs(horPlanes)<0.1] = 0.0 # we presume that cameras have been calibrated to a z=0 area
    Planes = np.zeros((locMax.size,4))
    Planes[:,2] = 1.0
    Planes[:,3] = horPlanes

    zRange = np.array((np.min(horPlanes),np.max(horPlanes)))
    Class = selectBestPlane(Pos,Cov,Planes,zRange)
    Lines, Scores = drawClassLines(Pos,Class)

    # look for bad classes, large overlaps
    Overlaps0 = findOverlaps(Pos,Lines)
    
    badOnes = np.argwhere(np.tril(Overlaps0,-1) > 0.5)
    includeIt = np.ones(len(Lines),np.bool)
    for i in range(badOnes.shape[0]):
        if Scores[badOnes[i,0]][-1] < Scores[badOnes[i,1]][-1]:
            includeIt[badOnes[i,0]] = False
        else:
            includeIt[badOnes[i,1]] = False

    keepers = np.where(includeIt)[0]
    numClasses = keepers.size
    Lines2 = []
    for i in keepers:
        Lines2.append(Lines[i])
    Lines = Lines2
    Planes = Planes[keepers,:]
    horPlanes = horPlanes[keepers]

    interfaceLine,interfaceCount,interfaceInds = findInterfaces(Pos,Lines)

    Classes = Classify(Pos,Lines)
    interfaces = np.argwhere(interfaceCount>0)

    for i in range(interfaces.shape[0]):
        i1 = interfaces[i,0]
        i2 = interfaces[i,1]
        # see if we should insert a slopeing connecting surface
        # the normal to the interfaceLine should be the direction of the
        # gradiend of the slope
        iLine = interfaceLine[i1,i2,:]
        transX = Pos[:,0] * iLine[0] + Pos[:,1] * iLine[1] - iLine[2]
    
        indsToUse = np.where((np.abs(transX) < 0.8) * (numObs > 1))[0]
        _,Error0 = computePlaneDeviance(Planes,Pos[indsToUse,:])
    
        # need a point that is reliably on the line...
        if abs(iLine[0]) > 0.1:
            Point0 = np.array((iLine[2]/iLine[0], 0.0))            
        else:
            Point0 = np.array((0.0, iLine[2]/iLine[1]))
                    
        SlopesToTry = np.arange(-12,12.05,0.05)
        Heights = horPlanes[np.array((i1,i2))]
        HeightsToTry = np.mean(Heights) + np.linspace(-0.5,0.5,61) * (np.max(Heights) - np.min(Heights))
        bestError = Error0
        for k1 in range(SlopesToTry.size):
            for k2 in range(HeightsToTry.size):
                Planes2 = Planes.copy()
                Normal = np.append(iLine[0:2],SlopesToTry[k1])
                Normal /= np.linalg.norm(Normal)
                d = np.sum(Normal * np.append(Point0, HeightsToTry[k2]))
            
                Planes2 = np.vstack((Planes2,np.append(Normal,d)))
                _,Error = computePlaneDeviance(Planes2,Pos[indsToUse,:])
                if Error < bestError:
                    bestError = Error
                    bestSlope = SlopesToTry[k1]
                    Planes3 = Planes2.copy()
        if bestError < Error0:
            if abs(bestSlope) < 5:
                Planes = Planes3.copy()
                # this is probably also the right time to update the boundary lines for this interface
                Lines.append(np.zeros((2,3)))
                # first: intersection between i1 and this new plane: just solve for z

                for cnt, ii in enumerate(np.array((i1,i2))):
                    interfLine = Planes[ii,np.array((0,1,3))] - Planes[-1,np.array((0,1,3))] * Planes[ii,2] / Planes[-1,2]
                    classCenter = np.mean(Pos[np.where(Classes[:,ii]==1)[0],0:2],axis=0)
                    if classCenter[0] * interfLine[0] + classCenter[1] * interfLine[1] < interfLine[2]:
                        interfLine *= -1
                    # just for tidiness, normalize
                    interfLine /= np.linalg.norm(interfLine[0:2])  
                    if cnt == 0:
                        Ind = interfaceInds[i1,i2]
                    else:
                        Ind = interfaceInds[i2,i1]
                    if Ind > -1:
                        Lines[ii][Ind,:] = -interfLine
                    Lines[-1][cnt,:] = interfLine

    xBins = np.linspace(xLimsOuter[0]+stepSize/2,xLimsOuter[1]-stepSize/2,int((xLimsOuter[1]-xLimsOuter[0]) / stepSize))
    yBins = np.linspace(yLimsOuter[0]+stepSize/2,yLimsOuter[1]-stepSize/2,int((yLimsOuter[1]-yLimsOuter[0]) / stepSize))

    x,y = np.meshgrid(xBins,yBins)
    x = x.transpose().ravel()
    y = y.transpose().ravel()
    Classes = Classify(np.vstack((x,y)).transpose(),Lines)
    zMap = np.zeros(x.shape)
    Weights = np.zeros(x.shape)
    for i in range(Classes.shape[1]):
        inds = np.where(Classes[:,i] == 1)[0]
        Weights[inds] = 1.0
        zMap[inds] = (Planes[i,3] - Planes[i,0] * x[inds] - Planes[i,1] * y[inds]) / Planes[i,2]

    eMap = {
        "xBins": xBins,
        "yBins": yBins,
        "xLims": xLims,
        "yLims": yLims,
        "xLimsOuter": xLimsOuter,
        "yLimsOuter": yLimsOuter,
        "Z":zMap,
        "Weight":Weights,
        "weightThres": 0.5}

    f = open(parent_path+"/elevationMap.npz", "wb")
    pickle.dump(eMap, f)    
    f.close()

    sio.savemat(parent_path+"/elevationSolution.mat", {'zMap':zMap, 'xBins':xBins, 'yBins':yBins, 'totWeight':Weights,'Classes':Classes } )

    cv2.namedWindow('elevationMap', cv2.WINDOW_NORMAL)
    minVal = np.min(zMap)
    maxVal = np.max(zMap)
    cv2.imshow('elevationMap', (zMap.reshape((xBins.size,yBins.size)).transpose()-minVal)/(maxVal-minVal))
    key = cv2.waitKey()
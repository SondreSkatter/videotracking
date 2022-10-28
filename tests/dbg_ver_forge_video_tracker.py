import tracker_viz_mgr
import cv2
import numpy as np
import time
import human_detector_pose_est as human_detector
import posenet, posenet_helper, pose_analyzer
import scene_manager, camera
import video_channels
import os
import bgtracker
import GT_labeling_tool

trckCfg = {  # config options
    "targetPeopleDetectFPS": 5,
    "recordData": False, # record new data set.  if set, no algorithms will run other than capturing data at the right sampling frequency
    "runLiveGPUdetection": False,
    "num_GPU_workers": 1,
    "num_orchestrators": 2,
    "cacheDetResults": False,
    "dropFramesWhenBehind": False,  # if live capture we always may have to drop frames, but this flag, if True, will force frame dropping also when running recorded data
    "usePersistedStaticObjects": False,
    "detectStaticObjects": False,
    "doFaceID": False, 
    "runTrackerInSepProcess": False,
    # display and video options
    "doRendering": True,
    "detectLeftLuggage": False,
    "saveVideo": True,
    "showStaticObjects": True,
    "showVisitorMap": False,
    "showGT": 2, # 0: don't show, 1: show GT boxes, 2: color objects according to correct wrt GT
    "cycleDet": False, # for example, if two cameras, ping pong between them. Since GPU is the scarce compute resource
    "globalGT": [],
    "minNumSightings": 1,
    "rec_name": [],
    "liveCapture": False # will be updated based on the channel inputs
    }

runLabelingTool = False  # if true, it sets up a session to label the video in terms of people tracking
reRunPoseAnal = True
reRunColorExtraction = False
interactiveMode = False
pauseOnErrorFrames = False
hungarianScoring = True
testDetectionOnly = False  # means, we turn off the tracking functionality and only discover detection
if interactiveMode or pauseOnErrorFrames:
    trckCfg["doRendering"] = True
if runLabelingTool:
    trckCfg["globalGT"] = 1 # just a hack to make sure scenemanager saves positions etc...
    interactiveMode = True
 
startOnFrame = 0

if 1:
    recordingName, lastFrame = 'Mar10-2020-1538', 506 
    #recordingName, lastFrame, startOnFrame = 'Apr02-2020-1114', 88, 35  # Sync test Sondre only
    #recordingName, lastFrame = 'Apr02-2020-1221', 267  # Skatter family, crazy
    recordingName, lastFrame = 'Apr19-2020-1421', 109 # Basu family #1 (frame is dropped 109->110 for axis wall
    recordingName, lastFrame = 'Apr19-2020-1423', 262  # Basu family #2
    recordingName, lastFrame = 'Apr19-2020-1424', 100  # Basu family #3 (frame is dropped 100->101 for axis pole)
    recordingName, lastFrame = 'Apr19-2020-1425', 98  # Basu family #4 (frame is dropped 98->99 for axis pole)
    recordingName, lastFrame = 'Apr19-2020-1426', 97  # Basu family #5 (frame is dropped 97->98 for axis pole)
    #recordingName, lastFrame = 'Apr19-2020-1427', 98  # Basu family #6 (frame is dropped 98->99 for axis pole)

    recordingName, lastFrame = 'Apr20-2020-1019',700

    myChannels = [ video_channels.channel('axis-pole',recordingName), video_channels.channel('axis-wall',recordingName) ]

    try:
        import wildtrackGT
        trckCfg["globalGT"] = wildtrackGT.wildtrackGT( os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(myChannels[0].CapturePath)))) +
            '/GTbitsFolder/groundtruth_true_xy.txt', trckCfg["targetPeopleDetectFPS"],myChannels[0].sceneScope, hungarianScoring)  
    except:
        print("No ground truth available")
elif 1:
    hungarianScoring = False
    myChannels = [ video_channels.channel('wildtrack1','cache'), video_channels.channel('wildtrack2','cache'), video_channels.channel('wildtrack3','cache'), video_channels.channel('wildtrack4','cache'), video_channels.channel('wildtrack5','cache'),  video_channels.channel('wildtrack6','cache'),  video_channels.channel('wildtrack7','cache')]
    trckCfg["targetPeopleDetectFPS"] = 2.0          
    import wildtrackGT
    
    trckCfg["globalGT"] = wildtrackGT.wildtrackGT('sampledata/wildtrack/Wildtrack_dataset/groundtruth_true_xy.txt', trckCfg["targetPeopleDetectFPS"],myChannels[0].sceneScope, hungarianScoring)          
    lastFrame = 12#399 #12
    if testDetectionOnly:
        assert myChannels[0].hasGT, "Set use2DGT to True in video_channels.py"
elif 1:
    recordingName, lastFrame = 'May02-2019-1635', 200   # classic Forge video
    recordingName, lastFrame = 'Oct10-2019-0918', 200  # car challenge video
    if 0:
        recordingName = 'Dec17-2019-1321'  # on stage and with boxes
        trckCfg["detectLeftLuggage"] = True
        lastFrame = 1200#200
        startOnFrame = 770#750#1080#750
    #recordingName = 'Live'
    if trckCfg["recordData"]:
        recordingName = 'Live'
    myChannels = [ video_channels.channel('warrior-pole',recordingName), video_channels.channel('warrior-wall',recordingName) ]
elif 1:
    myChannels = [ video_channels.channel('oxford','cache')]
    trckCfg["detectStaticObjects"] = True # those mannequins in the store windows...
    trckCfg["minNumSightings"] = 2
    lastFrame = int(2295 / 5)

numChannels = len(myChannels)
trckCfg, myTracker = scene_manager.enforceConfigCoherence(trckCfg, video_channels, myChannels)

readImages = trckCfg["doRendering"] or trckCfg["detectLeftLuggage"] or reRunColorExtraction or runLabelingTool

if readImages: 
    cap = []
    for c in range(numChannels):
        cap.append(cv2.VideoCapture(myChannels[c].CapturePath))
    nativeFPS = cap[0].get(cv2.CAP_PROP_FPS)
    if runLabelingTool:        
        myVizMgr = GT_labeling_tool.GT_labeling_tool(myChannels, trckCfg, myChannels[0].sceneScope, myTracker.myCamMappers)     
        trckCfg["detectLeftLuggage"] = False
    elif trckCfg["doRendering"]:
        myVizMgr = tracker_viz_mgr.tracker_viz_mgr(myChannels, trckCfg, myTracker.myCamMappers, combinedDisplay = True, interactivePlot = interactiveMode)     

if reRunPoseAnal:
    poseSolver = pose_analyzer.poseSolver(myChannels[0].sceneMapPath)  

imgBuff = [None] * numChannels
detResults_all = [None] * numChannels
trackerResults_all = [None] * (lastFrame + 1)
imCount = 0
largestCountSoFar = -1
keepGoing = True
renderTime = 0.0

for c in range(numChannels):
    if myChannels[c].hasGT: 
        myChannels[c].myGT.adjustGTtoFrameRate(trckCfg["targetPeopleDetectFPS"])

if trckCfg["detectLeftLuggage"]:
    bgTrackers = [None] * numChannels
    cv2.namedWindow('background0', cv2.WINDOW_NORMAL)
    cv2.namedWindow('background1', cv2.WINDOW_NORMAL)
    for c in range(numChannels):
        bgTrackers[c] = bgtracker.bgtracker()
else:
    bgTrackers = None

if imCount < startOnFrame: 
    if readImages:
        for c in range(numChannels):
            cap[c].set(cv2.CAP_PROP_POS_FRAMES, startOnFrame)
    imCount = startOnFrame

while imCount <= lastFrame and keepGoing:
    print(imCount)
    correct_det2D = [None] * numChannels
    correct_det3D = None
    for c in range(numChannels):
        detResults_all[c] = human_detector.get_frame(imCount, os.path.dirname(myChannels[c].CapturePath)+'/cached_det_results/')

        imageOK = detResults_all[c]["detectionWasRun"] # for a recording frames can be dropped, iand then there is a zeroed out image

        if reRunPoseAnal and imageOK:
            detResults_all[c] ["includeIt"], detResults_all[c] ["posAndHght"], detResults_all[c] ["posCov"], detResults_all[c] ["poseTypeProb"], \
                detResults_all[c] ["boxes"], detResults_all[c] ["Orientations"], detResults_all[c] ["pose_anal_time"], detResults_all[c] ["unitVectors"] = \
                pose_analyzer.report_detections(poseSolver, myTracker.myCamMappers[detResults_all[c] ["channelNum"]], \
                detResults_all[c] ["keypoint_coords"], detResults_all[c] ["scores"], detResults_all[c] ["keypoint_scores"], detResults_all[c] ["num"])
        if readImages and imageOK:
            frameNum = int(imCount * nativeFPS / trckCfg["targetPeopleDetectFPS"])
            cap[c].set(cv2.CAP_PROP_POS_FRAMES, frameNum)
            _, imgBuff[c] = cap[c].read()
            if reRunColorExtraction:
                 # rerun color extraction also
                detResults_all[c]["colorFeats"], detResults_all[c]["colorCov"], detResults_all[c]["hasColor"] = \
                        myTracker.poseHelper.computeColorFeatures(imgBuff[c], detResults_all[c]["keypoint_coords"], 
                        detResults_all[c]["keypoint_scores"], myTracker.numColorFeats)

            Clip = myChannels[c].clipRegion     
            if detResults_all[c]["detectionWasRun"]:
                if Clip is None: 
                    imgBuff[c] = cv2.resize(imgBuff[c],(0,0),fx=1.0/detResults_all[c]["output_scale"][1],fy=1.0/detResults_all[c]["output_scale"][0],interpolation = cv2.INTER_AREA)
                else:                    
                    imgBuff[c] = cv2.resize(imgBuff[c][Clip[0]:(Clip[2]+1),Clip[1]:(Clip[3]+1),:],(0,0),fx=1.0/detResults_all[c]["output_scale"][1],fy=1.0/detResults_all[c]["output_scale"][0],interpolation = cv2.INTER_AREA)

        if trckCfg["detectLeftLuggage"] and imageOK:
            bgTrackers[c].addFrame(imgBuff[c],detResults_all[c] ["boxes"], detResults_all[c]["output_scale"], detResults_all[c]["timeStamp"])

            winName = 'background'+str(c)
                #cv2.imshow(winName, bgTrackers[c].pixMeans* (np.expand_dims(bgTrackers[c].numObs,axis=2) > 0))
                #cv2.imshow(winName, bgTrackers[c].numObs / (imCount-startOnFrame+1))
                #cv2.imshow(winName, bgTrackers[c].pixMeansAlt * (np.expand_dims(bgTrackers[c].numObsAlt,axis=2) > 2))
                #cv2.imshow(winName, bgTrackers[c].opening*200)
            if 1:
                bgIm = bgTrackers[c].pixMeans.copy()
                #bgIm = bgTrackers[c].pixMeansAlt.copy()
                #bgIm = bgTrackers[c].pixMeansAlt.copy() * np.expand_dims(bgTrackers[c].numObsAlt > 0, axis = 2)
                for rr in range(bgTrackers[c].Boxes.shape[0]):
                    cv2.rectangle(bgIm,(int(bgTrackers[c].Boxes[rr,1]/bgTrackers[c].imScale[1]),\
                        int(bgTrackers[c].Boxes[rr,0]/bgTrackers[c].imScale[0])),\
                        (int(bgTrackers[c].Boxes[rr,3]/bgTrackers[c].imScale[1]),\
                        int(bgTrackers[c].Boxes[rr,2]/bgTrackers[c].imScale[0])),(0,0,255),2)
                cv2.imshow(winName,bgIm)
                
    if imCount > largestCountSoFar:
        if testDetectionOnly:
            # Make sure the gallery is empty, so that there is no tracking, just fresh detection
            myTracker.myTracks.slotUsed[myTracker.myTracks.Inds] = False
            myTracker.myTracks.nConseqMisses[myTracker.myTracks.Inds,:] = 0
            myTracker.myTracks.nConseqHits[myTracker.myTracks.Inds,:] = 0
            myTracker.myTracks.Inds = np.zeros(0,np.int) 
        if imCount == 200:
            d = 1
        trackerResults_all[imCount] = myTracker.report_frame(detResults_all, bgTrackers)
        largestCountSoFar = imCount

    trackerResults = trackerResults_all[imCount]
    trckErr = False
    if trckCfg["globalGT"] and not runLabelingTool and trackerResults["freshChannels"].size > 0:
        if testDetectionOnly:
            trckCfg["globalGT"].lastlLocIDmatch[:] = -1
        correct_det3D = trckCfg["globalGT"].reportResults(imCount,trackerResults["Pos"] ,trackerResults["Cov"], trackerResults["Inds"],trackerResults["numReincarnations"])
        trckErr = np.any(correct_det3D[0] == -1)
    for c in trackerResults["freshChannels"]:
        if myChannels[c].hasGT: 
            correct_det2D[c] = myChannels[c].myGT.reportResults(imCount,detResults_all[c]["boxes"][trackerResults["goodMatches"][c],:],trackerResults["Description"][c])

    if 0:
        # Saving tracks just to evaluate Kalman tracking filter
        import re
        personID = 1
        try:
            KalmanTrack
        except:
            KalmanTrack = np.zeros((0,2))
            MeasuredPos = [None] * 2
            MeasuredPos[0] = np.zeros((0,2))
            MeasuredPos[1] = np.zeros((0,2))

        KalmanTrack = np.vstack((KalmanTrack,myTracker.myTracks.posAndHeightTemp[personID,-1,0:2]))
        for Chan in range(2):
            matchIDs = np.zeros(len(trackerResults["Description"][Chan]),np.int)
            for iii in range(matchIDs.size):
                matchIDs[iii] = re.match('(\d{1,})([a-z])',trackerResults["Description"][Chan][iii]).group(1)     
            matched = trackerResults["goodMatches"][Chan][np.where(matchIDs == personID)[0]]
            MeasuredPos[Chan] = np.vstack((MeasuredPos[Chan],detResults_all[Chan]["posAndHght"][matched,0,0:2]))
            if imCount == 50:
                import scipy.io as sio
                sio.savemat('kalmandata.mat', {'KalmanTrack':KalmanTrack, 'MeasuredPos1':MeasuredPos[0], 'MeasuredPos2':MeasuredPos[1]} )            

    if trckCfg["doRendering"]:
        t0 = time.time()
        keepGoing = myVizMgr.renderFrame(imCount, trackerResults, detResults_all, imgBuff, correct_det2D, \
           keepGoing, correct_det3D=correct_det3D,haltFrame=(interactiveMode or (pauseOnErrorFrames and trckErr)))
        renderTime += time.time() - t0
        imCount += keepGoing  # this can move either forwards or backwards
        keepGoing = abs(keepGoing)
    else:
        trackerResults_all[imCount] = []
        imCount += 1
        
print('Time to render: {:.2f} per frame'.format(renderTime/(imCount+1)))

if readImages:
    for c in range(numChannels):
        cap[c].release()
if trckCfg["doRendering"] or runLabelingTool:
    myVizMgr.close()
if not runLabelingTool:
    if myChannels[0].hasGT:
        print(myChannels[0].myGT.getPerformanceSummary())
    if trckCfg["globalGT"]:
        print(trckCfg["globalGT"].getPerformanceSummary())        
myTracker.shutdown()    
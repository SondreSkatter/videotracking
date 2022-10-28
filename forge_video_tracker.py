# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
import sys
sys.path.append('utils')
sys.path.append('..\posenetpython')
sys.path.append('..\posenet-python')
import cam_capture
import tracker_viz_mgr
import multiprocessing
import cv2
import numpy as np
import time
import human_detector_pose_est as human_detector
import posenet_helper, pose_analyzer
import scene_manager, camera
import towncentreGT
import video_channels
import os
from custom_queue import Queue
from face_request_worker import FaceRequestWorkerPool

trckCfg = {  # config options
    "targetPeopleDetectFPS": 5,
    "recordData": True, # record new data set.  if set, no algorithms will run other than capturing data at the right sampling frequency
    "runLiveGPUdetection": True,
    "useTensorFlow": True,
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
    "showStaticObjects": True,
    "showVisitorMap": False,
    "showGT": 2, # 0: don't show, 1: show GT boxes, 2: color objects according to correct wrt GT
    "cycleDet": False, # for example, if two cameras, ping pong between them. Since GPU is the scarce compute resource
    "globalGT": [],
    "minNumSightings": 1,
    "rec_name": [],
    "liveCapture": False # will be updated based on the channel inputs
    }

hungarianScoring = True
lastFrame = -1
if 0:
    hungarianScoring = False
    myChannels = [ video_channels.channel('wildtrack1','cache'), video_channels.channel('wildtrack2','cache'), video_channels.channel('wildtrack3','cache'), video_channels.channel('wildtrack4','cache'), video_channels.channel('wildtrack5','cache'),  video_channels.channel('wildtrack6','cache'),  video_channels.channel('wildtrack7','cache')]
    trckCfg["targetPeopleDetectFPS"] = 2.0          
    import wildtrackGT
    trckCfg["globalGT"] = wildtrackGT.wildtrackGT('sampledata/wildtrack/Wildtrack_dataset/groundtruth_true_xy.txt', trckCfg["targetPeopleDetectFPS"],myChannels[0].sceneScope, hungarianScoring)          
    lastFrame = 12#399#12
elif 1:
    recordingName = 'Mar10-2020-1538'  # 
    recordingName = 'Apr02-2020-1221'  #
    recordingName = 'Apr19-2020-1421'  # Basu family #1
    recordingName = 'Apr19-2020-1423'  # Basu family #2
    recordingName = 'Apr19-2020-1424'  # Basu family #3
    recordingName = 'Apr19-2020-1425'  # Basu family #4
    recordingName = 'Apr19-2020-1426'  # Basu family #5
    recordingName = 'Apr19-2020-1427'  # Basu family #6

    recordingName = 'Apr20-2020-1019'
    #recordingName = 'Live'
    if trckCfg["recordData"]:
        recordingName = 'Live'
    myChannels = [ video_channels.channel('axis-pole',recordingName), video_channels.channel('axis-wall',recordingName) ]
    try:
        import wildtrackGT
        trckCfg["globalGT"] = wildtrackGT.wildtrackGT( os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(myChannels[0].CapturePath)))) +
            '/GTbitsFolder/groundtruth_true_xy.txt', trckCfg["targetPeopleDetectFPS"],myChannels[0].sceneScope, hungarianScoring)     
    except:
        pass
    #lastFrame = 200
elif 1:
    recordingName = 'May02-2019-1635'   # classic Forge video
    recordingName = 'Oct10-2019-0918'  # car challenge video
    #recordingName = 'Dec17-2019-1321'  # on stage and with boxes
    #recordingName = 'Live'
    if trckCfg["recordData"]:
        recordingName = 'Live'
    myChannels = [ video_channels.channel('warrior-pole',recordingName), video_channels.channel('warrior-wall',recordingName) ]
    lastFrame = 200
    #trckCfg["cycleDet"] = True
elif 1:
    myChannels = [ video_channels.channel('oxford','cache')]
    trckCfg["detectStaticObjects"] = True # those mannequins in the store windows...
    trckCfg["minNumSightings"] = 2
    lastFrame = int(2295 / 5)

numChannels = len(myChannels)

if __name__ == "__main__":
    if trckCfg["doFaceID"]:
        FacePool = FaceRequestWorkerPool(1)
        face_in_q = FacePool.input_queue
        face_out_q = FacePool.output_queue
    else:
        face_in_q = None
        face_out_q = None
    # enforcing some configuration constraints 
    trckCfg, myTracker = scene_manager.enforceConfigCoherence(trckCfg, video_channels, myChannels, face_in_q, face_out_q)
 
    print("Setting up capture streams...")
    captureMgr = cam_capture.capture_mgr(myChannels, trckCfg)

    if myTracker:
        print("Setting up GPU worker shop...")
        orchMgr = human_detector.orchestration_mgr(trckCfg["num_GPU_workers"], trckCfg["num_orchestrators"],trckCfg["cacheDetResults"], captureMgr.cap_msg_q, captureMgr.input_det_q, myTracker.track_in_q, myChannels, trckCfg["runLiveGPUdetection"], trckCfg["liveCapture"], scene_manager.numColorFeats, scene_manager.faceIDinterval, posenet_helper.posenet_helper(), myTracker.myCamMappers, trckCfg["cycleDet"], trckCfg["doRendering"], trckCfg["useTensorFlow"])
 
    if trckCfg["doRendering"]:
        myVizMgr = tracker_viz_mgr.tracker_viz_mgr(myChannels, trckCfg, myTracker.myCamMappers, combinedDisplay = True)    

    print("Waiting for cameras to sync...")
    if captureMgr.synchronize_streams():
        print("Done with the lengthy initialization. Ready to roll now!")
    else:
        print("Couldnt initialze streams. Exiting now.")
        sys.exit(0)
    print('Det Queue size: {:d}'.format(captureMgr.input_det_q[0].qsize()))

    imCount = -1
    channelToDisplay = 0
    imgBuff = [None] * numChannels
    detResults_all = [None] * numChannels
    correct_det3D = None

    totTimeA = 0.0
    totTimeB = 0.0
    totTimeTracking = 0.0
    totTimeD = 0.0
    totTimeE = 0.0
    timeStart = time.time()

    if trckCfg["recordData"]:
        captureMgr.monitorRecording()

    keepGoing = True and myTracker
    lastFrameInMovie = -1
    droppedFrames = 0
    nOutputsWaiting = np.zeros(numChannels, np.int)
    while keepGoing:
        imCount += 1
        if imCount == 1:
            # jst to take out the latency from the frame time estimates
            timeStart = time.time()
            totTimeB = 0.0

        time00 = time.time()
        lastFrameInMovie = captureMgr.reachedLastFrame()

        if lastFrameInMovie > 0:
            print("end of movie reached. Just finishing the frames in the pipeline... last frame is: "+str(lastFrameInMovie))
            if imCount >= lastFrameInMovie:
                if imCount == lastFrameInMovie:
                    keepGoing = False
                else:
                    imCount -= 1  # just to get the stats right
                    break
        # Main capture loop
        print('Frame: {:d}, elapsed time: {:.2f}.'.format(imCount,time00-timeStart))
       
        time11 = time.time()
        totTimeA += time11 - time00

        # get the detection data from previous frame (just wait until it's done)
        # need to make sure the data we sent to the tracker is time ordered

        frame_status = np.zeros(numChannels, np.int)
        
        numLoops = 0
        while np.any(frame_status == 0): 
            for c in np.where(frame_status == 0)[0]:
                if not orchMgr.output_det_q[c].empty():
                    imgBuff[c], detResults_all[c] = orchMgr.output_det_q[c].get()
                    nOutputsWaiting[c] = orchMgr.output_det_q[c].qsize()
                    if detResults_all[c]["detectionWasRun"]:
                        frame_status[c] = 1
                        print('Ch: {:d}, Det. time: {:.3f}, Prep time: {:.3f}, Prep time2: {:.3f}, pose anal. time: {:.3f}. Det I/O q size: {:d}  {:d}'.format(c,detResults_all[c]["proc_time"],detResults_all[c]["prepTime"],detResults_all[c]["prepTime2"],detResults_all[c]["pose_anal_time"],nOutputsWaiting[c],captureMgr.input_det_q[c].qsize()))
                        #print('frame Num: {:d}, channel: {:d}. output queue size {:d}'.format(imCount, c, orchMgr.output_det_q[c].qsize()))
                    else:
                        print('Detection no run on frame Num: {:d}, channel: {:d}.'.format(imCount, c))
                        if imgBuff[c] == []:
                            print("Frame dropped. Gotta keep up!")
                            droppedFrames += 1
                        frame_status[c] = 3        
            numLoops += 1
            if numLoops > 2:
                lastFrameInMovie = captureMgr.reachedLastFrame()
                if lastFrameInMovie > -1 and lastFrameInMovie < imCount:
                    print("end of movie reached. Just finishing the frames in the pipeline... last frame is: "+str(lastFrameInMovie))
                    keepGoing = False
                    break
                time.sleep(0.005)
  
        #print('Time stamp in channel 0: {:.2f}'.format( detResults_all[0]["timeStamp"]))
        begTrack = time.time()
        totTimeB += begTrack - time11

        trackerResults = myTracker.report_frame(detResults_all)

        print('Time to track: ',trackerResults["trackTime"])
        time22 = time.time()
        totTimeTracking += trackerResults["trackTime"]

        # now, what's left is scoring and rendering.. We'll choose to do this on the last completed frame
        correct_det2D = [None] * numChannels
        for c in trackerResults["freshChannels"]:
            if myChannels[c].hasGT: 
                correct_det2D[c] = myChannels[c].myGT.reportResults(imCount,detResults_all[c]["boxes"][trackerResults["goodMatches"][c],:],trackerResults["Description"][c])
        if trckCfg["globalGT"] and trackerResults["freshChannels"].size > 0:
            correct_det3D = trckCfg["globalGT"].reportResults(imCount,trackerResults["Pos"] ,trackerResults["Cov"], trackerResults["Inds"],trackerResults["numReincarnations"])
        
        time33 = time.time()
        totTimeD += time33 - time22
        if trckCfg["doRendering"]:
            if (not trckCfg["liveCapture"]) or (np.max(nOutputsWaiting) < 2) or (imCount % 5 == 0):
                # the idea is to waste less resources on rendering if falling behind
                keepGoing = myVizMgr.renderFrame(imCount, trackerResults, detResults_all, imgBuff, correct_det2D,
                    keepGoing, correct_det3D=correct_det3D)

        if imCount == lastFrame and not trckCfg["liveCapture"]:
            keepGoing = False 

        totTimeE += time.time() - time33

    if myTracker: 
        print("time per frame: ",(time.time()-timeStart)/(imCount+0),", tracking time per view: ", totTimeTracking/(imCount+1)) 
        print('Percentage dropped frames: {:.1f}'.format(100 * droppedFrames/((imCount+1) * numChannels)))
    captureMgr.shutdown()

    if trckCfg["doRendering"]: 
        myVizMgr.close()

    if myTracker: 
        if myChannels[channelToDisplay].hasGT:
            print(myChannels[channelToDisplay].myGT.getPerformanceSummary())
        if trckCfg["globalGT"]:
            print(trckCfg["globalGT"].getPerformanceSummary())        
        myTracker.shutdown()    
        orchMgr.shutdown()
        print('TimeA: {:.3f}, TimeB: {:.3f}, TimeC: {:.3f}, TimeD: {:.3f}, TimeE: {:.3f}'.format(totTimeA/(imCount+1),totTimeB/(imCount+0),totTimeTracking/(imCount+1),totTimeD/(imCount+1),totTimeE/(imCount+1)))
    if trckCfg["doFaceID"]:
        FacePool.shutdown(1)

    time.sleep(0.3) # give them some time to shut down
    print("Number of active processes still (main): ",len(multiprocessing.active_children()))


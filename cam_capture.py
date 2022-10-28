import multiprocessing
from custom_queue import Queue
import numpy as np
import time, cv2, os, platform

numSyncMsgs = 7  # send tjhis number of messages to each camera for time syncing
timeIntervals = 0.07# this amount of time between each sync message
addedDelaySec = 0.8 # post mean time for sync events, add this number before we define time 0

def prepare_recording(Folder, recording_name, targetPeopleDetectFPS, imSize):
    # this function is called when someone wants to record data 
    sampleFolder = Folder + 'sampleFolder'
    newDir = sampleFolder+"/"+recording_name
    os.makedirs(newDir)
    recording_folder = newDir
        
    if platform.system() == 'Windows':
        fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
    else:
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    #fourcc = cv2.VideoWriter_fourcc('H', 'F', 'Y', 'U')
    recording_file_name = newDir + "/" + recording_name + '.mp4'
    recording = cv2.VideoWriter(recording_file_name,fourcc, targetPeopleDetectFPS, (int(imSize[1]),int(imSize[0])))
    return recording

def capture_worker(cap_input_q, msg_out_q, output_cap_q, channelNum, channelName, capturePath, Folder, imSize, trckCfg, max_q_size):
    # First: set up capture
    slowDownToRealTime = trckCfg["doRendering"] and not trckCfg["liveCapture"]
    dropFramesWhenBehind = trckCfg["dropFramesWhenBehind"]
    targetPeopleDetectFPS = trckCfg["targetPeopleDetectFPS"]
    liveCapture = trckCfg["liveCapture"]
    doRecording = trckCfg["recordData"] and (trckCfg["rec_name"] != [])
    if doRecording:
        recording = prepare_recording(Folder, trckCfg["rec_name"], trckCfg["targetPeopleDetectFPS"], imSize)
    
    frameInfo = {
        "channelNum": channelNum,
        "channelName": channelName
        }        
    waterMark = int(max_q_size / 2)
        
    t0before = time.time()
    cap = cv2.VideoCapture(capturePath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps: ",fps)
    if not cap.isOpened() or fps > 1000:
        print("Could not open stresm: {:d}.".format(channelNum))
        msg_out_q.put((channelNum,"failed"))
        return
    t0after = time.time()
    initTime = t0after - t0before
    
    detectFreq = fps / trckCfg["targetPeopleDetectFPS"]  # note: not necessarily an integer number

    if liveCapture: 
        print("Cam started: stream {:d} , start time: {:.2f}, after that time: {:.2f}".format(channelNum,t0after-t0before,time.time()-t0after))

        smallestDiff = 1000.0
        runnerUpDiff = smallestDiff
        # Catch up with a backlog of frames on the camera buffer from initiation
        while True:
            cap.read() # just make sure we move off the last frame if available
            camTime = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 
            thisTimeStamp = time.time() - t0before
            thisDiff = thisTimeStamp - camTime

            if thisDiff < smallestDiff:
                runnerUpDiff = smallestDiff
                smallestDiff = thisDiff
            elif thisDiff < runnerUpDiff:
                runnerUpDiff = thisDiff
            
            print("pre cam sync: stream {:d}, this Diff: {:.2f}, smallest Diff: {:.2f}, sys time: {:.2f}, cam time: {:.2f}".format(channelNum,thisDiff, smallestDiff, thisTimeStamp, camTime))  
            
            if camTime > 2.5 * initTime and runnerUpDiff - smallestDiff < 0.02:
                print("Determined cam sync: stream {:d} , sys time: {:.2f}, cam time: {:.2f}".format(channelNum,smallestDiff, camTime))  
                timeDiff = smallestDiff + t0before
                break
            time.sleep(0.001)
    else:
        framesLost = 0

    msg_out_q.put((channelNum,"ready"))

    # Now wait for start signal from manager
    while True:
        if not cap_input_q.empty():
            message, timeStart = cap_input_q.get()
            if message == "start":                     
                if liveCapture:
                    print("Start signal received: stream {:d} , sys time: {:.2f}, cam time: {:.2f}".format(channelNum,time.time()-timeDiff,cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 ))
                    time0 = timeStart - timeDiff  # time0 is the start time for teh synchronized streams, imn units of the camera relative time
                else:
                    imCount = -1
                break
        if liveCapture:
            cap.read() # just make sure we move off the last frame if available
            time.sleep(0.001)
        else:
            time.sleep(0.01)

    frameCount = -1

    while True:        
        if not cap_input_q.empty():
            message = cap_input_q.get()
            if message == "die":
                print(time.time(),'Stream shutting down. Processed {:d} frames'.format(frameCount))
                cap.release()
                if doRecording:
                    recording.release()

        if not liveCapture and slowDownToRealTime:
            # just to avoid rendering going too fast
            thisTime = time.time()
            if (thisTime - timeStart) * fps < imCount:
                time.sleep(0.01)
                continue
           
        r, imgBuff = cap.read()  

        if r:
            sendIt = False
            if liveCapture:
                #timestamp = time.time() - time0
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - time0

                thisFrameNum = int(np.floor(timestamp * targetPeopleDetectFPS))
                if thisFrameNum > frameCount:
                    framesLost = thisFrameNum - frameCount - 1
                    if framesLost > 0:                        
                        print("Lost {:d} frames from camera. This wont end well".format(thisFrameNum - frameCount - 1))
                    frameCount = thisFrameNum
                    sendIt = True
                    #time.sleep(0.2) # Just to create choking
                    if frameCount % 10 == 0:
                        print(time.time(),', ',timestamp,', Stream {:d} processed {:d} frames'.format(channelNum,frameCount))
            else:
                imCount += 1
                if int(imCount % detectFreq) == 0:
                    frameCount += 1 
                    timestamp = frameCount / targetPeopleDetectFPS
                    sendIt = True
            if sendIt:
                # had to make this weird copy because of an issue in Queue that will send a duplicate of the item in the 
                # qeue if the object changes soon after the put() https://bugs.python.org/issue8037
                frameInfoToSend = frameInfo.copy()
                frameInfoToSend["timeStamp"] = timestamp
                frameInfoToSend["frameNum"] = frameCount 
                if doRecording:
                    tt0 = time.time()
                    for misses in range(framesLost):
                        recording.write(imgBuff*0)
                    recording.write(imgBuff)
                    #print("Stream {:d}, timestamp: {:.2f}, Time to save frame: {:.2f}".format(channelNum, timestamp,time.time()-tt0))
                    if frameCount % 5 == 0:
                        output_cap_q.put((frameInfoToSend,imgBuff))  
                else:
                    for misses in range(framesLost):
                        missedFrameInfoToSend = frameInfo.copy()
                        missedFrameInfoToSend["timeStamp"] = 0.0
                        missedFrameInfoToSend["frameNum"] = frameCount + misses - framesLost
                        output_cap_q.put((missedFrameInfoToSend,[]), block=True)    
                    if dropFramesWhenBehind:
                        qSize = output_cap_q.qsize()
                        if qSize >= waterMark:
                            if np.random.random() > 0.5 * (1.0 - (qSize - waterMark) / (max_q_size - waterMark)):
                                imgBuff = []   
                    elif np.mean(imgBuff) < 1.0:
                        # A zeroed out image was added in the recorded movie because frames were dropped
                        print("Dropped frame in recording detected")
                        imgBuff = []
                    output_cap_q.put((frameInfoToSend,imgBuff), block=True)     
                    #print("Putting to work frame: "+str(frameInfoToSend["frameNum"]) + ", queue size is: "+str(output_cap_q.qsize()))
                    
        elif not liveCapture:
            # end of the movie?
            msg_out_q.put(("end of movie",frameCount))


class capture_mgr:
    def __init__(self, Channels, trckCfg):
        self.cap_msg_q = []
        self.cap_msg_out_q = Queue()
        self.cap_pool = []
        self.input_det_q = []
        self.numChannels = len(Channels)
        self.liveCapture = trckCfg["liveCapture"]
        max_q_size = 4 # this will not be enforced for the queue, but we will start to drop frames above this number
        for c in range(self.numChannels):
            if self.liveCapture :
                self.input_det_q.append(Queue())
            else:
                self.input_det_q.append(Queue(2))
            self.cap_msg_q.append(Queue())
            self.cap_pool.append(multiprocessing.Pool(1, capture_worker, (self.cap_msg_q[c],self.cap_msg_out_q,self.input_det_q[c], c,Channels[c].camName, Channels[c].CapturePath, Channels[c].Folder, Channels[c].imSize, trckCfg, max_q_size)))  
            if Channels[c].hasGT: 
                Channels[c].myGT.adjustGTtoFrameRate(trckCfg["targetPeopleDetectFPS"])

    def shutdown(self):
        for c in range(self.numChannels):
            self.cap_msg_q[c].put("die")

        time.sleep(0.25)
        for c in range(self.numChannels):
            while not self.input_det_q[c].empty():
                self.input_det_q[c].get()
            self.cap_pool[c].terminate()

    def synchronize_streams(self):
        numReadyChannels = 0
        #camTime0 = -np.ones(self.numChannels)
        while numReadyChannels < self.numChannels:
            chanNum, message = self.cap_msg_out_q.get()
            if message == "failed":
                return False
            numReadyChannels += 1
        print("All camera have reported ready, sending start signal now")

        timeNow = time.time()
        for c in range(self.numChannels):
            self.cap_msg_q[c].put(("start",timeNow + 0.1))
        return True

    def reachedLastFrame(self):
        lastFrameInMovie = -1
        if not self.cap_msg_out_q.empty():
            msg = self.cap_msg_out_q.get()
            if msg[0] == "end of movie":
                lastFrameInMovie = msg[1]
        return lastFrameInMovie

    def monitorRecording(self):
        cv2.namedWindow('cameraview', cv2.WINDOW_NORMAL)
        keepGoing = True
        channelSwitchFreq = 5
        channelToDisplay = 0
        counter = 0
        while keepGoing:
            for c in range(self.numChannels):
                if not self.input_det_q[c].empty():
                    frameInfo,imgBuff = self.input_det_q[c].get()
                    counter += 1
                    channelToDisplay = (counter // (self.numChannels * channelSwitchFreq)) % self.numChannels
                    if c == channelToDisplay:
                        cv2.imshow('cameraview', imgBuff)
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord('q'):
                            cv2.destroyWindow('cameraview')
                            keepGoing = False
import multiprocessing
from custom_queue import Queue
import posenet_helper
import pose_analyzer
import numpy as np
import time, cv2
import pickle
import os

scaleByAbsoluteSize = True
if scaleByAbsoluteSize:
    targetSize = (int(0.6*1080),int(0.6*1920))
else:
    scale_factor = 0.6 # scale the input image with this amount, for performance

target_num_pixels = 0.65 * 0.65 * 1080 * 1920

output_stride = 16  # an attribute of the posenet model


def get_frame(frameNum, Path):
    try:
        f = open(Path + "poseresults"+str(frameNum), "rb")
        detResult = pickle.load(f)
        f.close() 
        return detResult
    except:
        print('No cached available, please run with useLiveGPU turned on first.')
        return []

def valid_resolution(width, height, output_stride):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height

def scale_image_for_pose_model(source_img):   

    if 1:
        targetSize = np.array((source_img.shape[0], source_img.shape[1]),np.float) * np.sqrt(target_num_pixels / (source_img.shape[0]*source_img.shape[1]))
        target_width, target_height = valid_resolution(
            targetSize[1], targetSize[0], output_stride=output_stride)
        #print("target_width: ",target_width,"target_height: ",target_height,", source_img size: ",source_img.shape)
    else:
        if scaleByAbsoluteSize:
            target_width, target_height = valid_resolution(
            targetSize[1], targetSize[0], output_stride=output_stride)
        else:
            target_width, target_height = valid_resolution(
                source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
        
    output_scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])    
    return cv2.resize(source_img, (target_width, target_height), interpolation = cv2.INTER_LINEAR), output_scale

def GPU_worker(input_GPU_q, output_GPU_q):
    import tensorflow as tf
    import posenet
    # Load a (frozen) Tensorflow model into memory.
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']

        while True:
            for q in range(len(input_GPU_q)):
                if not input_GPU_q[q].empty():
                    frame = input_GPU_q[q].get(block=True)
                    if len(frame) == 2:
                        start_time = time.time()
                        jobInfo = frame[0]
                        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                            model_outputs,
                            feed_dict={'image:0': frame[1].astype(np.float32) * (2.0 / 255.0) - 1.0}
                        )
                        jobInfo["proc_time"] = time.time() - start_time
                        output_GPU_q[q].put((jobInfo, heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result))
                    elif len(frame) == 1 and frame[0] == "die":
                        break
            time.sleep(0.001) # SS: I added this
        sess.close()

def GPU_worker_pyTorch(input_GPU_q, output_GPU_q):
    # Import model
    from pathlib import Path
    # Deep HRNet config
    pytorch_cfg_file = Path('deep_hrnet_yolov3/inference-config-coco.yaml')
    pytorch_model_path = Path('deep_hrnet_yolov3/models/pose_coco/pose_hrnet_w32_384x288.pth')
    yolo_config = Path('deep_hrnet_yolov3/lib/yolov3/config/yolov3.cfg')
    yolo_weights = Path('deep_hrnet_yolov3/lib/yolov3/weights/yolov3.weights')
    conf_thres = 0.8
    yolo_nms_thres = 0.4
    use_yolo = False

    from deep_hrnet_yolov3.inference_yolo import DeepHRNet
    # initialize model
    hrnet = DeepHRNet(str(yolo_config), str(yolo_weights), str(pytorch_cfg_file),
                      str(pytorch_model_path), use_yolo)
    # run
    while True:
        for q in range(len(input_GPU_q)):
            if not input_GPU_q[q].empty():
                frame = input_GPU_q[q].get(block=True)
                if len(frame) == 2:
                    start_time = time.time()
                    jobInfo = frame[0]
                    # run model on image in frame[0]
                    box_scores, keypoint_coords, keypoint_scores = \
                        hrnet.run(frame[1].squeeze(), conf_thres, yolo_nms_thres)
                    jobInfo["proc_time"] = time.time() - start_time
                    # returmnn: pose_scores, keypoint_scores, keypoint_coords 
                    output_GPU_q[q].put((jobInfo, box_scores, keypoint_scores, keypoint_coords))
                elif len(frame) == 1 and frame[0] == "die":
                    break
        time.sleep(0.001)  # SS: I added this

def pose_est_orchestrator(det_comm_q,cap_msg_q,input_q,pose_est_q,pose_result_q,output_q, output_trckr_q, saveOutput, channels, useLiveGPU, liveCapture, numColorFeats, faceIDinterval, poseHelper, camMappers, cycleDet, passAlongImage, useTensorFlow, useEmptyWorker=False):
    if useTensorFlow:
        import posenet
    numChannels = len(channels)
    Min_score = posenet_helper.det_threshold #0.05  # another threshold being applied downstream

    Paths = [None] * numChannels
    if saveOutput or not useLiveGPU:        
        for c in range(numChannels):
            Paths[c] = os.path.dirname(channels[c].CapturePath)+'/cached_det_results/'
            try:
                os.mkdir(Paths[c] ) 
            except:
                pass

    myWorkerNum = -1
    while myWorkerNum == -1:
        if not det_comm_q.empty():
            #the initial message we're expecting is to be told which worker number we are...
            myWorkerNum, myChannels = det_comm_q.get()
        time.sleep(0.001)

        myPoseSolver = pose_analyzer.poseSolver(channels[0].sceneMapPath)  

    nextChannelNumLoc = 0
    nextChannelNum = myChannels[0]
    numChannelsLoc = myChannels.size
    jobsPending = 0
    imgMap = {}
    imgMapScaled = {}
    awaitingJobs = {}
    jobsAwaiting = 0
    runTrackerInSepProcess = (output_trckr_q != [])

    while True:
        if jobsPending < 3:
            if not input_q[nextChannelNum].empty():
                time0 = time.time()
                frame = input_q[nextChannelNum].get()
                if len(frame) == 2:
                    jobInfo = frame[0]                    
                    assert nextChannelNum == jobInfo["channelNum"], 'unexpected channel number received...'
                    jobInfo["channelNumLoc"] = nextChannelNumLoc

                    nextChannelNumLoc = (nextChannelNumLoc + 1) % numChannelsLoc
                    nextChannelNum = myChannels[nextChannelNumLoc]
                    
                    thisIm = frame[1]
                    #print('Taking job for channel: {:d}, frame {:d}'.format(jobInfo["channelNum"],jobInfo["frameNum"]))
                    #print("Jobs pending: "+ str(jobsPending) + ", waiting in queue: " + str(input_q[jobInfo["channelNum"]].qsize()) + ", out qeue size: "+str(output_q[jobInfo["channelNum"]].qsize()))
                    
                    processFrame = useLiveGPU
                    if thisIm == []:
                        processFrame = False
                        input_image = []
                        #print('Receiving empty job for channel: {:d}, frame {:d}'.format(jobInfo["channelNum"],jobInfo["frameNum"]))
                    else:
                        Clip = channels[jobInfo["channelNum"]].clipRegion
                        
                        # prepping the image for pose detector
                        if Clip is None: 
                            input_image, jobInfo["output_scale"] = scale_image_for_pose_model(thisIm)  
                            jobInfo["clipOffset"] = np.zeros(2,np.int)
                        else:                            
                            input_image, jobInfo["output_scale"] = scale_image_for_pose_model(thisIm[Clip[0]:(Clip[2]+1),Clip[1]:(Clip[3]+1),:])  
                            jobInfo["clipOffset"] = Clip[0:2]

                    mapKey = str(jobInfo["channelNum"]) + 's' + str(jobInfo["frameNum"])
                    if cycleDet:
                        if jobInfo["channelNum"] != (jobInfo["frameNum"] % numChannels):
                            # don't process this frame. Just pass the image along
                            processFrame = False

                    if processFrame:
                        jobsPending += 1                        
                        imgMap[mapKey] = thisIm
                        imgMapScaled[mapKey] = input_image    
                        jobInfo["prepTime"] = time.time() - time0
                        jobInfo["detectionWasRun"] = True
                        # send to pose detector
                        pose_est_q[myWorkerNum].put((jobInfo,np.expand_dims(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), axis=0)))
                    else:
                        if not passAlongImage:
                            input_image = []
                        if useLiveGPU:
                            jobInfo["detectionWasRun"] = False
                            if jobsPending > 0:
                                # need to hold off on sending this job, otherwise it will be out of order
                                awaitingJobs[mapKey] = jobInfo
                                imgMapScaled[mapKey] = input_image
                                jobsAwaiting += 1
                                #print('Awaiting empty job for channel: {:d}, frame {:d}'.format(jobInfo["channelNum"],jobInfo["frameNum"]))
                                #print("Jobs pending: "+ str(jobsPending) + ", waiting in queue: " + str(input_q[jobInfo["channelNum"]].qsize()) + ", out qeue size: "+str(output_q[jobInfo["channelNum"]].qsize()))
                                #print("pose_result_q size: "+ str(pose_result_q[myWorkerNum].qsize()) + ", pose_est_q size: " + str(pose_est_q[myWorkerNum].qsize()))
                            else:
                                if runTrackerInSepProcess:
                                    output_trckr_q[jobInfo["channelNum"]].put(jobInfo)
                                output_q[jobInfo["channelNum"]].put((input_image,jobInfo))

                                #print('Bypassing empty job for channel: {:d}, frame {:d}'.format(jobInfo["channelNum"],jobInfo["frameNum"]))
                        else:
                            # used cached pose detection results
                            jobInfo = get_frame(jobInfo["frameNum"], Paths[jobInfo["channelNum"]])
                            assert "detectionWasRun" in jobInfo.keys(), 'Cached results is out of date, please run with useLiveGPU turned on first.'
                            if runTrackerInSepProcess:
                                output_trckr_q[jobInfo["channelNum"]].put(jobInfo)                            
                            output_q[jobInfo["channelNum"]].put((input_image,jobInfo))

        if useLiveGPU:
            if not pose_result_q[myWorkerNum].empty():
                # receive results from the pose detector, prep the output
                time0 = time.time()
                pose_results = pose_result_q[myWorkerNum].get()    
                detResult = pose_results[0]
                if useTensorFlow:                    
                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                        pose_results[1].squeeze(axis=0),
                        pose_results[2].squeeze(axis=0),
                        pose_results[3].squeeze(axis=0),
                        pose_results[4].squeeze(axis=0),
                        output_stride=output_stride,
                        max_pose_detections=45,
                        min_pose_score=Min_score)

                else:
                    pose_scores = pose_results[1]
                    keypoint_scores = pose_results[2]
                    keypoint_coords = pose_results[3]           
                
                keypoint_coords *= detResult["output_scale"]                
                keypoint_coords += detResult["clipOffset"] 
                    
                goodOnes = np.where(pose_scores > Min_score)[0]

                detResult["scores"] = pose_scores[goodOnes]
                detResult["keypoint_scores"] = keypoint_scores[goodOnes,:]
                detResult["keypoint_coords"] = keypoint_coords[goodOnes,:,:]
                detResult["num"] = goodOnes.size
                mapKey = str(detResult["channelNum"])+ 's' + str(detResult["frameNum"])
                thisIm = imgMap[mapKey]

                detResult["colorFeats"], detResult["colorCov"], detResult["hasColor"] = \
                    poseHelper.computeColorFeatures(thisIm, detResult["keypoint_coords"], 
                    detResult["keypoint_scores"], numColorFeats)

                if detResult["frameNum"] % faceIDinterval == 0:
                    N = detResult["num"]
                    detResult["faceOpportunity"] = np.zeros(N, np.bool)
                    detResult["faceChip"] = [None] * N 

                    for i in range(N):
                        detResult["faceOpportunity"][i], faceBox = poseHelper.getFaceArea(detResult["keypoint_coords"][i,:,:], detResult["keypoint_scores"][i,:], thisIm.shape)
                        if detResult["faceOpportunity"][i]:                            
                            detResult["faceChip"][i] = thisIm[faceBox[0]:faceBox[2],faceBox[1]:faceBox[3],:]
                                
                detResult["prepTime2"] = time.time() - time0

                detResult["includeIt"], detResult["posAndHght"], detResult["posCov"], detResult["poseTypeProb"], \
                    detResult["boxes"], detResult["Orientations"], detResult["pose_anal_time"], detResult["unitVectors"] = \
                    pose_analyzer.report_detections(myPoseSolver, camMappers[detResult["channelNum"]], \
                    detResult["keypoint_coords"], detResult["scores"], detResult["keypoint_scores"], detResult["num"])
                
                if not passAlongImage:
                    imgMapScaled[mapKey] = []
                if runTrackerInSepProcess:
                    output_trckr_q[detResult["channelNum"]].put(detResult)
                output_q[detResult["channelNum"]].put((imgMapScaled[mapKey],detResult))

                jobsPending -= 1

                if jobsAwaiting > 0:
                    fr = detResult["frameNum"]
                    chLoc = detResult["channelNumLoc"]                  
                    while True:
                        chLoc = (chLoc + 1) % numChannelsLoc
                        if chLoc == 0:
                            fr += 1
                        ch = myChannels[chLoc]
                        mapKey2 = str(ch) + 's' + str(fr)
                        if mapKey2 in awaitingJobs.keys():
                            if runTrackerInSepProcess:
                                output_trckr_q[ch].put(awaitingJobs[mapKey2])                            
                            output_q[ch].put((imgMapScaled[mapKey2],awaitingJobs[mapKey2]))
                            if saveOutput:                    
                                f = open(Paths[ch]+"poseresults"+str(fr), "wb")
                                pickle.dump(awaitingJobs[mapKey2], f)
                                f.close()
                            imgMapScaled.pop(mapKey2)
                            awaitingJobs.pop(mapKey2)
                            jobsAwaiting -= 1
                        else:
                            break

                # release the image from memory now
                imgMap.pop(mapKey)
                imgMapScaled.pop(mapKey)  

                if saveOutput:                    
                    f = open(Paths[detResult["channelNum"]]+"poseresults"+str(detResult["frameNum"]), "wb")
                    pickle.dump(detResult, f)
                    f.close()

        time.sleep(0.001) # SS: I added this

class orchestration_mgr:
    def __init__(self, num_GPU_workers, num_orchestrators,cacheDetResults, cap_msg_q, input_det_q, tracker_in_q, Channels, runLiveGPUdetection, liveCapture, numColorFeats, faceIDinterval, PosenetHelper, camMappers, cycleDet, doRendering, useTensorFlow):
        self.numChannels = len(Channels)
        self.pose_est_q = [] 
        self.pose_result_q = []        
        self.output_det_q = []
        self.output_trckr_q = tracker_in_q
        self.det_comm_q = Queue()

        for c in range(self.numChannels):
            self.output_det_q.append(Queue(2))

        self.num_GPU_workers = num_GPU_workers
        num_orchestrators = min(num_orchestrators, self.numChannels)
        self.runLiveGPUdetection = runLiveGPUdetection
        if runLiveGPUdetection:
            for i in range(num_orchestrators):
                self.pose_est_q.append(Queue()) 
                self.pose_result_q.append(Queue()) 
            if useTensorFlow:
                self.gpu_pool = multiprocessing.Pool(num_GPU_workers, GPU_worker, (self.pose_est_q,self.pose_result_q))          
            else:
                self.gpu_pool = multiprocessing.Pool(num_GPU_workers, GPU_worker_pyTorch, (self.pose_est_q,self.pose_result_q))   
        else:
            num_orchestrators = 1
        self.det_pool = multiprocessing.Pool(num_orchestrators, pose_est_orchestrator, (self.det_comm_q,cap_msg_q,input_det_q,self.pose_est_q,self.pose_result_q,self.output_det_q, self.output_trckr_q, cacheDetResults, Channels, runLiveGPUdetection, liveCapture, numColorFeats, faceIDinterval, PosenetHelper, camMappers, cycleDet, doRendering, useTensorFlow))  
        for c in range(num_orchestrators):
            # need to pass message to the workers so they know which number they are (decides the queue)
            # need to split up the work load as well
            channelsToCover = np.where(np.arange(self.numChannels) % num_orchestrators == c)[0]
            self.det_comm_q.put((c,channelsToCover))   
        if runLiveGPUdetection:
            # instead of an init we need to process the first image, before we can kick everything loose...
            for c in range(self.numChannels):
                jobInfo = {
                    "channelNum": c,
                    "frameNum": int(-1),
                    "channelName": "dummy",
                    "timeStamp": time.time()
                    }
                input_det_q[c].put((jobInfo, np.zeros(Channels[c].imSize,np.uint8)))   

            for c in range(self.numChannels):      
                self.output_det_q[c].get() # flush that first image
        
    def shutdown(self):
        if self.runLiveGPUdetection:
            for i in range(len(self.pose_est_q)):
                while not self.pose_est_q[i].empty():
                    self.pose_est_q[i].get()
                self.pose_est_q[i].put(("die"))
                while not self.pose_result_q[i].empty():
                    self.pose_result_q[i].get()
            self.gpu_pool.terminate()
        for c in range(self.numChannels):
            while not self.output_det_q[c].empty():
                self.output_det_q[c].get()

        self.det_pool.terminate()

import os, cv2, time, camera, numpy as np
import video_channels
import human_detector_pose_est, posenet_helper, pose_analyzer
import scene_manager
import pickle
import os
import matplotlib.pyplot as plt
import scipy.io as sio

# 1) verifies correctness compared to a defined baseline result, and
# 2) analyzes the compute performance

establish_new_baseline = False

myChannels = [ video_channels.channel('wildtrack1','cache'), video_channels.channel('wildtrack2','cache'), video_channels.channel('wildtrack3','cache'), video_channels.channel('wildtrack4','cache'), video_channels.channel('wildtrack5','cache'),  video_channels.channel('wildtrack6','cache'),  video_channels.channel('wildtrack7','cache')]

numFrames = 40
parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(myChannels[0].Folder)), os.pardir))

numChannels = len(myChannels)
poseSolver = pose_analyzer.poseSolver(myChannels[0].sceneMapPath)

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

numPostures = len(poseSolver.myPoseHelper.postures)
numAllocItems = 5000
allocSteps = 1000
N = 0
Pos = np.zeros((numAllocItems,numPostures+1,3))
Cov = np.zeros((numAllocItems,numPostures+1,3,3))
Probs = np.zeros((numAllocItems,numPostures))
Orientations = np.zeros((numAllocItems,scene_manager.numAngleBins))

timeBegin = time.time()
fr = 0
for fr in range(numFrames):   
    if fr % 25 == 0:
        print("frame: " + str(fr))
    for c in range(numChannels):
        detResult = human_detector_pose_est.get_frame(fr, Paths[c])
        if detResult["detectionWasRun"]:
            BBs = poseSolver.getBBfromPose(detResult["keypoint_scores"], detResult["keypoint_coords"])  
            for i in np.where(detResult["scores"] > 0.25)[0]:              
                heightInPixels = BBs[i,2] - BBs[i,0]
                success, Pos[N,:,:], Cov[N,:,:,:], Probs[N,:], Orientations[N,:], _ = poseSolver.get_person_coords_from_keypoints(detResult["keypoint_coords"][i,:,:], detResult["scores"][i], detResult["keypoint_scores"][i,:], heightInPixels, cameras[c])
                if success:
                    N += 1
                if N >= numAllocItems:
                    numAllocItems += allocSteps
                    Pos.resize((numAllocItems,numPostures+1,3))
                    Cov.resize((numAllocItems,numPostures+1,3,3))
                    Probs.resize((numAllocItems,numPostures))
                    Orientations.resize((numAllocItems,scene_manager.numAngleBins))
    fr += 1
    
timePerPoseMsec = (time.time() - timeBegin) / N * 1000
print(str(N) + ' poses procesed, time per pose analyze: {:.2f} msec, per frame: {:.1f} msec'.format(timePerPoseMsec, 1000*(time.time() - timeBegin) / (fr * numChannels)))
Pos = Pos[0:N,:,:]
Cov = Cov[0:N,:,:,:]
Probs = Probs[0:N,:]
Orientations = Orientations[0:N,:]

if establish_new_baseline:
    f = open(parent_path+"/PoseBaseline.npz", "wb")
    pickle.dump(Pos, f)
    pickle.dump(Cov, f)
    pickle.dump(Probs, f)
    pickle.dump(Orientations, f)    
    pickle.dump(timePerPoseMsec, f)
    f.close()
    print("Baseline has been established and saved.")
else:
    # compare to baseline
    f = open(parent_path+"/PoseBaseline.npz", "rb")
    Pos0 = pickle.load(f)
    Cov0 = pickle.load(f)
    Probs0 = pickle.load(f)
    Orientations0 = pickle.load(f) 
    timePerPoseMsec0 = pickle.load(f) 
    f.close()
    print("")
    print("Comparing to baseline...")
    print('Time per pose analyze: {:.3f} msec, vs {:.3f} msec for baseline'.format(timePerPoseMsec, timePerPoseMsec0))
    N0 = Probs0.shape[0]
    if N0 != N:
        print('Different number of objects compared to baseline... {:d} vs {:d} objects current vs baseline.'.format(N,N0))
    else:
        posDiff = Pos - Pos0
        avePosDiff = np.sum(np.abs(posDiff)) / N

        print('Average abs pos difference per pose: {:.10f}'.format(avePosDiff))
        aveProbDiff = np.sum(np.abs(Probs-Probs0)) / N
        print('Average abs prob difference per pose: {:.10f}'.format(avePosDiff))
        print('Average abs orientation difference per pose: {:.10f}'.format(np.sum(np.abs(Orientations-Orientations0)) / N))

import sys
sys.path.append('.')
sys.path.append('utils')
sys.path.append('../posenetpython')
import tensorflow
import multiprocessing
from custom_queue import Queue
import numpy as np, cv2
import time
import human_detector_pose_est as human_detector
import posenet, posenet_helper
import video_channels

# The idea is to only measure the timing for the pose estimation, which is done on the GPU
maxNumWorkers = 2
completionTime = np.zeros(maxNumWorkers)
numImsInTest = 100
myChannels = [ video_channels.channel('oxford','cache')]
cap = cv2.VideoCapture(myChannels[0].CapturePath)
frameCounter = 0

numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if __name__ == "__main__":

    input_det_q = [Queue()]
    output_det_q = [Queue()]

    for numWorkers in range(1,maxNumWorkers+1):
        print('Testing with {:d} workers'.format(numWorkers))
        det_pool = multiprocessing.Pool(numWorkers, human_detector.GPU_worker, (input_det_q,output_det_q))  
        # instead of an init we need to process the first image, before we can kick everything loose...
        input_image, _ = human_detector.scale_image_for_pose_model(np.zeros(myChannels[0].imSize,np.uint8))  
        jobInfo = {"dummy":0}
        input_det_q[0].put((jobInfo, np.expand_dims(input_image,axis=0))) 
        #input_det_q.put((0, int(0),0.0) + posenet_helper.scale_image_for_pose_model(np.zeros(myChannels[0].imSize,np.uint8)))   
        while (output_det_q[0].empty()):
            time.sleep(0.01)        
        output_det_q[0].get() # flush that first image

        # ready to start the test now
        jobsSent = 0
        jobsCompleted = 0
        startTime = time.time()
        while jobsCompleted < numImsInTest:
            while jobsSent < numImsInTest and input_det_q[0].qsize() < 5:
                   # add another job
                   r, img = cap.read()
                   timestamp = time.time()
                   input_image, _ = human_detector.scale_image_for_pose_model(img)  
                   input_det_q[0].put((jobInfo, np.expand_dims(input_image,axis=0))) 
                   frameCounter = (frameCounter + 1) % numFrames
                   if frameCounter == 0:
                       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)               
                   jobsSent += 1
            # check the output queue now
            while not output_det_q[0].empty():
                output_det_q[0].get()
                jobsCompleted += 1
                if jobsCompleted % 10 == 0:
                    print('{:d} of {:d} jobs completed.'.format(jobsCompleted,numImsInTest))
            time.sleep(0.001)
        completionTime[numWorkers-1] = time.time() - startTime
        det_pool.terminate()

    cap.release()

    print('Test results for {:d} images:'.format(maxNumWorkers))
    for numWorkers in range(maxNumWorkers):
        print('With {:d} workers pose detection took {:.2f} seconds, {:.3f} seconds per frame.'.format(numWorkers+1,completionTime[numWorkers], completionTime[numWorkers] / numImsInTest))
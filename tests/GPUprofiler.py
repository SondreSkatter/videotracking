import sys
sys.path.append('.')
sys.path.append('utils')
sys.path.append('../posenetpython')
import tensorflow as tf
import numpy as np
import time, cv2
import human_detector_pose_est as human_detector
import posenet, posenet_helper
import video_channels

# The idea is to only measure the timing for the pose estimation, which is done on the GPU
numImsInTest = 100
myChannels = [ video_channels.channel('oxford','cache')]
cap = cv2.VideoCapture(myChannels[0].CapturePath)
frameCounter = 0

numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Load a (frozen) Tensorflow model into memory.
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']

    for fr in range(numImsInTest):
        r, img = cap.read()
        timestamp = time.time()
        input_image, _ = human_detector.scale_image_for_pose_model(img)  

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
            model_outputs,
            feed_dict={'image:0': np.expand_dims(input_image,axis=0).astype(np.float32) * (2.0 / 255.0) - 1.0}
                        )
        if fr % 10 == 0:
            print('{:d} of {:d} jobs completed.'.format(fr,numImsInTest))


    cap.release()

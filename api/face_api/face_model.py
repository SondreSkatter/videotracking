import cv2
import dlib
from face_api.utils.time_op import time_op

#brought the face model to the API directory 
class FaceModel():
    predictor_model = './models/face_id/shape_predictor_68_face_landmarks.dat'
    face_rec_model_path = './models/face_id/dlib_face_recognition_resnet_model_v1.dat'
    num_face_dims = 128

    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_pose_predictor = dlib.shape_predictor(self.__class__.predictor_model)
        self.facerec = dlib.face_recognition_model_v1(self.__class__. face_rec_model_path)

    def execute(self, img):
        upSampleFactor = 2
        detSensitivity = 0.5#0.0
        minFaceHeightPixels = 37
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceHeightPixels = 0
        detected_face,scores,idx = time_op('face_detection_time', lambda: self.face_detector.run(gray, upSampleFactor, detSensitivity))
        if len(detected_face) == 1:
            thisRect = dlib.rectangle(int(detected_face[0].left()),int(detected_face[0].top()),int(detected_face[0].right()),int(detected_face[0].bottom()))   
            faceHeightPixels = thisRect.bottom() - thisRect.top()
            if scores[0] > detSensitivity: # should always be true at this point
                if  faceHeightPixels >= minFaceHeightPixels:
                    pose_landmarks = time_op('shape_predictor_time', lambda: self.face_pose_predictor(img, thisRect))
                    newFaceData = time_op('face_recognition_time', lambda: self.facerec.compute_face_descriptor(img, pose_landmarks))
                    return newFaceData, scores[0], faceHeightPixels
        return [], scores, faceHeightPixels

import numpy as np
import cv2

det_threshold = 0.15
PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]
CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

class posenet_helper:
    def __init__(self):

        self.det_threshold = 0.25 # it may seem strange to have this value higher than the one used initially by detection but it makes a difference when 
        # dealing with duplicates
        self.det_threshold_large_objects = 0.25 
        self.how_large_is_large = 1.0  # height  of person, in pixels, to the image height 
        self.when_does_large_begin = 0.5  # height  of person, in pixels, to the image height 

        self.PART_NAMES = [
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
        ]


        self.numParts = len(self.PART_NAMES)
        self.flippedInds = np.array((0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15),np.int)
        #self.Parts_For_Color_extraction = ["Front Torso","Rear Torso","Face","Upper Arms"]
        #self.Parts_For_Color_extraction = ["Front Torso","Rear Torso"]
        self.KLtoP1eqs = np.array([[[0.0, 0.3],[0.45, 1.0],[1.5,3.0]],[[0.0, 0.3],[0.45, 1.0],[1.5,3.0]]]) # ["Front Torso","Rear Torso"]
        #self.KLtoP1eqs = np.array([[[0.0, 0.5],[0.95, 1.0],[6,1.35]],\
        #    [[0.0, 0.55],[0.95, 1.0],[4,2.0]]]) # ["Front Torso","Rear Torso"]
        self.KLtoP1eqs = np.array([[0.0, 0.5],[0.95, 1.0],[6,1.35]])
        self.KLtoP1eqs = np.array([[0.0, 0.5],[0.95, 1.0],[6,2.0]])
        self.KLtoP1eqs = np.array([[0.0, 0.2],[0.3, 1.0],[1.6,3.0]])
        self.KLtoP1eqs = np.array([[0.0, 0.05],[1.0, 1.0],[3.5,3.5]]) # for 2D and 1D
        #self.KLtoP1eqs = np.array([[0.0, 0.05],[0.85, 1.0],[3.3,4]]) # for 2D and 1D
        #self.KLtoP1eqs = np.array([[0.0, 0.05],[0.85, 1.0],[0.95, 1.0],[3.5,3.5]]) # for 2D and 1D

        #self.KLtoP1eqs = np.array([[0.7, 0.08],[2., 1.],[5.7,3.2]]) # for 3D
        #self.Parts_For_Color_extraction = []

        #self.numBodyPartsForCol = len(self.Parts_For_Color_extraction)

        self.nominalPersonHeight = 1.65
        self.varPersonHeight = np.square(0.35) 
        self.humanHeightLimits = np.array([0.8, 2.8]) # min max for height of a human...
        
        # relative height proportions
        bottomToAnkle = 0.039

        eyesToTop = 0.078
        noseEarsToTop = 0.09756
        bottomToElbow = 0.5854
        bottomToWrist = 0.439

        bottomToKnee = 0.26       
        bottomToHip = 0.53
        bottomToShoulders = 0.82

        bottomToNoseEars = 1.0 - noseEarsToTop
        bottomToEyes = 1.0 - eyesToTop

        self.postures = ["Standing", "Sitting", "Sitting on ground", "Lieing", "Bending over"]
        self.postureMobile = np.array((True,True,False,False,True))
        self.relHeights = np.zeros((self.numParts,len(self.postures)), np.float)
        self.relHeights[:,0] = np.array([bottomToNoseEars,bottomToEyes,bottomToEyes,bottomToNoseEars,bottomToNoseEars,bottomToShoulders,
                                    bottomToShoulders,bottomToElbow,bottomToElbow,bottomToWrist,bottomToWrist,
                                    bottomToHip,bottomToHip,bottomToKnee,bottomToKnee,bottomToAnkle,bottomToAnkle])
        self.relHeights[:,1] = self.relHeights[:,0] # sitting
        self.relHeights[0:13,1] -=  bottomToHip - bottomToKnee
        self.relHeights[0:13,2] = self.relHeights[0:13,0] - bottomToHip # "Sitting on ground"
        self.relHeights[0:13,4] = bottomToHip #+ bottomToShoulders) / 2# bending over
        self.relHeights[13:,4] = self.relHeights[13:,0] 

        self.nominalWidths = np.array([0.0,-0.03,0.03,-0.12,0.12,-0.2,0.2,-0.2,0.2,-0.35,0.35,-0.25,0.25,-0.2,0.2,-0.2,0.2])
        # horVars describes offset from human axis, including when the axis is off from a vertical axis
        self.nominalHorVars = np.zeros((self.numParts,len(self.postures)), np.float)
        self.nominalHorVars[:,:] = np.square(np.array([[0.3,0.3,0.3,0.35,0.35,0.35,0.35,0.4,0.4,0.4,0.4,0.2,0.2,0.3,0.3,0.35,0.35]]).transpose())        
        self.nominalHorVars[13:,1:3] = np.square(0.4) #Sitting & sitting on ground        
        self.nominalHorVars[0:7,4] = np.square(np.array((0.5,0.5,0.5,0.5,0.5,0.4,0.4))) #bending over        
        self.nominalHorVars[:,3] = np.square(np.array([0.5,0.5,0.5,0.5,0.5,0.4,0.4,0.3,0.3,0.2,0.2,0.2,0.2,0.35,0.35,0.5,0.5])) #lieing down

        # vertVars includes both 1) error in pose detection, 2) deviation from nominal anatomy, and 3) deviation from straight standing posture
        #self.vertVars = np.square(np.array([0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]))  
        self.vertVars = np.square(1*np.array([0.2,0.2,0.2,0.2,0.2,0.25,0.25,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]))  

        self.torsoCoords = np.array([5, 6, 11, 12],np.int)

        self.ratioTorsoWidthToHeight = 2.0 * np.mean(abs(self.nominalWidths[self.torsoCoords])) / self.nominalPersonHeight
        self.ankleInds = np.array([15, 16],np.int)
        self.headInds = np.arange(5)
        self.shoulderInds = np.array([5, 6],np.int)
        self.hipInds = np.array([11, 12], np.int)
        self.kneeInds = np.array([13, 14],np.int)
        self.indsForHeightEst = np.concatenate((self.ankleInds,self.kneeInds,self.hipInds,self.shoulderInds,self.headInds))
    
    def heightInSpec(self, Height, heightVar):
        heightDelta = 0. * np.sqrt(heightVar)
        return np.logical_and(Height+heightDelta > self.humanHeightLimits[0], Height-heightDelta < self.humanHeightLimits[1])

    def scoreTooLow(self, heightRatio, score):
        # larger humans tends to get undetected with this pose detector, so we're allowing smaller scores for these
        if heightRatio < self.when_does_large_begin:
            thres = self.det_threshold
        elif heightRatio > self.how_large_is_large:
            thres = self.det_threshold_large_objects
        else:
            thres = self.det_threshold_large_objects + (self.det_threshold - self.det_threshold_large_objects) * (self.how_large_is_large - heightRatio) / (self.how_large_is_large - self.when_does_large_begin)
        return score < thres

    def getPolygonForColors(self, coords, imSize, keypoint_score):
        isGood = False
        # check quality            
        Scores = keypoint_score[self.torsoCoords]
        if np.sum((Scores > 0.35).astype(np.int)) >= self.torsoCoords.size - 1 and np.sum((Scores > 0.15).astype(np.int)) == self.torsoCoords.size:
            isGood = True
        return isGood, coords[np.array((5,6,12,11)),:]  

    def getNumPixelsTorso(self,coords):

        Width = 0.5*abs(coords[5,1] - coords[6,1] + coords[11,1] - coords[12,1])
        Height = 0.5*abs(coords[11,0] - coords[5,0] + coords[12,0] - coords[6,0])
        return Width * Height

    def getHeightWeights(self, keypoint_score, coords):
        goodInds = self.indsForHeightEst
        return keypoint_score[goodInds], self.relHeights[goodInds,:], self.nominalWidths[goodInds], self.nominalHorVars[goodInds,:], coords[goodInds,:], self.vertVars[goodInds]

    def getFaceArea(self, coords, keypoint_score, imSize):
        if np.sum((keypoint_score[self.headInds] > 0.85).astype(np.int)) >= self.headInds.size - 1:
            # add another criterion: eyes symmetric around the nose
            if abs(np.log(np.linalg.norm(coords[1,:]-coords[0,:])) - np.log(np.linalg.norm(coords[2,:]-coords[0,:]))) < np.log(1.3):
                if not np.any(coords[self.headInds,:] < 0) and not np.any(coords[self.headInds,0] > imSize[0]-1) and not np.any(coords[self.headInds,1] > imSize[1]-1):
                    Margin = int(0.5 * (np.max(coords[self.headInds,1]) - np.min(coords[self.headInds,1])))
                    box = np.array([max(0,np.min(coords[self.headInds,0])-2 * Margin), max(0,np.min(coords[self.headInds,1])-Margin), 
                                    min(imSize[0]-1,np.max(coords[self.headInds,0]) + 2 * Margin), min(imSize[1],np.max(coords[self.headInds,1])+Margin)],np.int)
                    return True, box
        return False, -1

    def compareKeypoints2(self, coordsA, coordsB, scoresA, varsB):
        # need to come up with a weight metric that uses scores for the new ponits and var for the database points
        Weights = 1.0 / (np.square(np.linalg.norm(varsB, axis=1)) + 1.0)
        Norms = np.linalg.norm(coordsA - coordsB, axis=1)  / np.sqrt(np.abs(coordsA[6,1]-coordsA[5,1]) * np.abs(coordsB[6,1]-coordsB[5,1]))
        combNorm = np.sum(Norms * Weights) / np.sum(Weights)
        return combNorm
    
    def detectDuplicates(self, scores, coords, part_score, Overlaps):   
        N = scores.size
        sqrtScales = np.zeros(N,np.float)
        includeIt =  np.ones(N,np.bool)
        suspectedDupe = np.zeros((N,N),np.int)

        for i in range(N):
            sqrtScales[i] = np.sqrt(np.abs(np.mean(coords[i,self.hipInds,0]) - np.mean(coords[i,self.shoulderInds,0])))
        for i in range(N):
            for j in np.where(Overlaps[i,0:i] > 0.0)[0]:
                Norms = np.linalg.norm(coords[i,:,:] - coords[j,:,:], axis=-1) 
                NormsRev = np.linalg.norm(coords[i,self.flippedInds,:] - coords[j,:,:], axis=-1)  
                if np.sum(NormsRev) < np.sum(Norms):
                    Norms = NormsRev
                Norms /= sqrtScales[i] * sqrtScales[j]

                Weights = part_score[i,:] * part_score[j,:]
                weightSame = np.sum(Weights[Norms < 0.15])
                weightDifferent = np.sum(Weights[Norms > 0.25])

                if weightSame / np.sum(Weights)  > 0.1:# and weightDifferent / np.sum(Weights)  < 0.5:# 0.7:#5:#0.05: 
                    suspectedDupe[i,j] = 1
                    suspectedDupe[j,i] = 1

        # analyze the topologies: if a person has two duplicates and each of the duplicates only have one then remove the connector
        numConnections = np.sum(suspectedDupe == 1,axis=0)
        multiConnector = np.where(numConnections > 1)[0]

        for i in multiConnector[np.argsort(numConnections[multiConnector])[::-1]]:
            friends = np.where(suspectedDupe[i,:] == 1)[0]
            if friends.size > 1: # might have changed since the start
                if np.any(np.sum(suspectedDupe[friends,:] == 1,axis=1) < friends.size):
                    # we can remove this twin
                    suspectedDupe[i,friends] = 0
                    suspectedDupe[friends,i] = 0
                    includeIt[i] = False
                    # The remainers should inherit the score from their removed connector person, if it's higher
                    lowerScores = np.where(scores[friends]-scores[i] < 0.0)[0]
                    scores[friends[lowerScores]] = scores[i]

        # then, pull them out along the order of the scores
        while True:
            hasDupes = np.where(np.sum(suspectedDupe,axis=0) > 0)[0]
            if hasDupes.size == 0:
                break
            oneToRemove = hasDupes[np.argmin(scores[hasDupes])]
            suspectedDupe[oneToRemove,:] = 0
            suspectedDupe[:,oneToRemove] = 0
            includeIt[oneToRemove] = False
        return includeIt

    def computeColorFeatures(self, img, coords,  keypointscores, numColorFeats):
        N = coords.shape[0]
        colorFeats = np.zeros((N, numColorFeats))
        colorCov = np.zeros((N, numColorFeats,numColorFeats))
        hasColor = np.zeros(N, np.bool)

        for i in range(N):
            hasColor[i], polygon = self.getPolygonForColors(coords[i,:,:], img.shape, keypointscores[i,:])
            if hasColor[i]:
                polygon = polygon.astype(np.int)
                Rect = np.array((max(0,np.min(polygon[:,0])),max(0,np.min(polygon[:,1])),min(np.max(polygon[:,0])+1,img.shape[0]-1),min(np.max(polygon[:,1])+1,img.shape[1]-1)))
                if Rect[2] - Rect[0] < 2 or Rect[3] - Rect[1] < 2:
                    hasColor[i] = False
                    continue

                im2 = img[Rect[0]:Rect[2],Rect[1]:Rect[3],:].copy().astype(np.float)/ 255.0 + 0.0001    
                if numColorFeats < 3:
                    if numColorFeats == 1:
                        # just use gray scale
                        im2 = np.expand_dims(np.sum(im2,axis=2),axis=2)
                    elif 0:
                        # 1D "chromatic" feature 
                        im2 = im2[:,:,0:2] / (np.expand_dims(np.sum(im2,axis=2),axis=2)) + 0.0001
                        im2 = np.expand_dims(im2[:,:,0] / np.sum(im2,axis=2),axis=2)
                    elif numColorFeats == 2:
                        # compute lighting invariant color feature
                        im2 = im2[:,:,0:2] / (np.expand_dims(np.sum(im2,axis=2),axis=2))
                polygon -= np.expand_dims(Rect[0:2],axis=0)
                Mask = np.zeros((Rect[2]-Rect[0],Rect[3]-Rect[1]), np.uint8)
                #showImage('Mask',Mask)
                cv2.fillConvexPoly(Mask, polygon[:,::-1], 255)

                if 0:
                    im3 = img.copy()
                    poly2 = polygon.copy() + np.expand_dims(Rect[0:2],axis=0)
                    cv2.fillConvexPoly(im3, poly2[:,::-1], (0,0,255))
                    showImage('im3',im3)

                numPixels = np.sum(Mask) / 255
                if numPixels < 30:
                    hasColor[i] = False
                    continue
                im2 = np.reshape(im2,(Mask.size, numColorFeats))
                im2 = im2[Mask.ravel().astype(np.bool),:]
                colorFeats[i,:] = np.mean(im2,axis=0)
                colorCov[i,:,:] = np.cov(im2.transpose()) + 1E-10 * np.eye(numColorFeats)
        return colorFeats, colorCov, hasColor


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):

    results = []
    for left, right in CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results

def draw_skel_and_kp(
        out_img, instance_scores, keypoint_scores, keypoint_coords, Color, 
        min_pose_score=0.5, min_part_score=0.5):
    # re-implemented this from utility in posenet because it created additional copies of image
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    cv2.drawKeypoints(
        out_img, cv_keypoints, out_img, color=Color,#(255, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=Color) #(255, 255, 0))

def draw_single_skel_and_kp(
        out_img, score, keypoint_scores, keypoint_coords, Color, 
        min_pose_score=0.5, min_part_score=0.5, forIndexImage = False):
    # re-implemented this from utility in posenet because it created additional copies of image
    cv_keypoints = []

    if score < min_pose_score:
        return

    adjacent_keypoints = get_adjacent_keypoints(
        keypoint_scores, keypoint_coords, min_part_score)

    for ks, kc in zip(keypoint_scores, keypoint_coords):
        if ks < min_part_score:
            continue
        cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    cv2.drawKeypoints(
        out_img, cv_keypoints, out_img, color=Color,
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    if forIndexImage:
        cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=Color, thickness = 6, lineType = 4) 
    else:
        cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=Color, thickness = 1) 


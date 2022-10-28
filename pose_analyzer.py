import cv2, time, camera
import numpy as np
import posenet_helper
import scene_manager
import pickle

class elevation_mapper:
    def __init__(self, elevMapPath):
        if elevMapPath != []:
            f = open(elevMapPath, "rb")
            self.Map = pickle.load(f)
            f.close()
            self.xBinsSize = self.Map["xBins"][1] - self.Map["xBins"][0]
            self.yBinsSize = self.Map["yBins"][1] - self.Map["yBins"][0]
            self.Nx = self.Map["xBins"].size
            self.Ny = self.Map["yBins"].size
            self.N = self.Nx * self.Ny
            self.center_x = np.mean(self.Map["xLimsOuter"])
            self.center_y = np.mean(self.Map["yLimsOuter"])
            self.hasMap = True
        else:
            self.hasMap = False

    def lookupHeight(self,x,y):
        if not self.hasMap:
            return True, 0.0
        if not self.isInScope(x,y):
            return False, 0.0
        else:
            # Determine the index into the height map
            xInd = int((x - self.Map["xLimsOuter"][0])/self.xBinsSize)
            yInd = int((y - self.Map["yLimsOuter"][0])/self.yBinsSize)
            Ind = yInd + self.Ny * xInd
            assert Ind >= 0 and Ind < self.N, "index out of range... shouldnt happen though."
            if self.Map["Weight"][Ind] >= self.Map["weightThres"]:
                return True, self.Map["Z"][Ind]
            else:
                return False, 0.0

    def isInScope(self, x,y):
        return (x > self.Map["xLimsOuter"][0]) * (x < self.Map["xLimsOuter"][1]) * (y > self.Map["yLimsOuter"][0]) * (y < self.Map["yLimsOuter"][1])

    def isInCoreScope(self, x,y):
        return (x > self.Map["xLims"][0]) * (x < self.Map["xLims"][1]) * (y > self.Map["yLims"][0]) * (y < self.Map["yLims"][1])

    def getLineTransect(self,x0,y0,dirVec,Spacing=0.5):
        if not self.hasMap:
            return False

        xTransects = np.zeros(2)
        yTransects = np.zeros(2)
        tTransect = np.zeros(2)
        addedTs = 0
        if abs(dirVec[1]) > 0.00000001:
            # compute transects to horizontal edges
            t = (self.Map["yLimsOuter"] - y0) / dirVec[1]
            xTransects = x0 + t[0:2] * dirVec[0]
            for i in range(2):
                if xTransects[i] > self.Map["xLimsOuter"][0] and xTransects[i] < self.Map["xLimsOuter"][1]:
                    tTransect[addedTs] = t[i]
                    addedTs += 1

        if abs(dirVec[0]) > 0.00000001:
            # compute transects to vertical edges
            t = (self.Map["xLimsOuter"] - x0) / dirVec[0]
            yTransects = y0 + t * dirVec[1]
            for i in range(2):
                if yTransects[i] > self.Map["yLimsOuter"][0] and yTransects[i] < self.Map["yLimsOuter"][1]:
                    tTransect[addedTs] = t[i]
                    addedTs += 1

        assert tTransect.size == 2, "Intersections are not right...."        

        Range = np.arange(np.min(tTransect)+Spacing,np.max(tTransect),Spacing)
        if 1:
            Ind = self.Ny * ((x0 + Range * dirVec[0] - self.Map["xLimsOuter"][0])/self.xBinsSize).astype(np.int) + \
                ((y0 + Range * dirVec[1] - self.Map["yLimsOuter"][0])/self.yBinsSize).astype(np.int)
        else:
            Points = np.array([[x0],[y0]]) + Range * np.expand_dims(dirVec,axis=1)
            # Determine the index into the height map
            xInd = ((Points[0,:] - self.Map["xLimsOuter"][0])/self.xBinsSize).astype(np.int)
            yInd = ((Points[1,:] - self.Map["yLimsOuter"][0])/self.yBinsSize).astype(np.int)
            Ind = yInd + self.Ny * xInd

        #assert np.all(Ind >= 0) and np.all(Ind < self.N), "index out of range... shouldnt happen though."

        goodInds = np.where(self.Map["Weight"][Ind] >= self.Map["weightThres"])[0]
        if goodInds.size == 0:
            return np.zeros(0,np.int), 0.0

        return Range[goodInds], self.Map["Z"][Ind[goodInds]]



class poseSolver:
    def __init__(self, elevMapPath):
        self.doOrientation = scene_manager.doOrientation
        self.numAngleBins = scene_manager.numAngleBins

        self.elevMap = elevation_mapper(elevMapPath)

        self.myPoseHelper = posenet_helper.posenet_helper()
        self.numPoseTypes = len(self.myPoseHelper.postures)
        self.goodInds = self.myPoseHelper.indsForHeightEst
        self.relHeights = self.myPoseHelper.relHeights[self.goodInds,:]
        self.nomWidths = self.myPoseHelper.nominalWidths[self.goodInds]
        self.horVars = self.myPoseHelper.nominalHorVars[self.goodInds,:]
        self.vertVars = self.myPoseHelper.vertVars[self.goodInds]
        N = self.goodInds.size
        self.MatforBB = np.ones((N,2),np.float)
        self.MatforBB [:,1] = -self.relHeights[:,0]
        self.numPostures = len(self.myPoseHelper.postures)
        self.angleBins = []

        # setting the matrix values that are always the same
        self.mainMat = np.zeros((N*2+1,3), np.float)
        self.mainb = np.zeros((N*2+1,1), np.float)
        self.mainW = np.zeros((N*2+1,1), np.float)

        self.mainMat[0:N,0] = 1.0
        self.mainMat[N:(2*N),1] = 1.0
        self.mainMat[-1,2] = 1.0
        self.mainb[-1,0] = self.myPoseHelper.nominalPersonHeight   # height of a average person?
        self.mainW[-1,0] = 1.0 / self.myPoseHelper.varPersonHeight # this is the uncertainty of height... +- 35 cm


        # make it a set of 2D problems: each item represents a different pose (standing, sitting, etc)
        self.poseMat = np.ones((N+1,2), np.float)
        self.poseb = np.zeros((N+1,1), np.float)
        self.poseW = np.zeros((N+1,1), np.float)

        self.poseMat[-1,0] = 0.0
        self.poseb[-1,0] = self.myPoseHelper.nominalPersonHeight   # height of a average person?
        self.poseW[-1,0] = 1.0 / self.myPoseHelper.varPersonHeight # this is the uncertainty of height... +- 35 cm

        # make it a set of 2D problems: each item represents a different pose (standing, sitting, etc)
        self.poseMat3D = np.ones((N+1,3), np.float)
        self.poseMat3D[-1,0:2] = 0.0
        self.cov3D = np.zeros((3,3), np.float)
        self.cov3D[2,2] = 1.0E10
        self.sol3Db = np.zeros((3,1), np.float)

        self.Sol2a = np.zeros((self.numPostures+1,2,1),np.float)
        self.Cov2a = np.zeros((self.numPostures+1,2,2),np.float)
        self.torsoInds = np.where(np.logical_and(self.relHeights[:,0] > 0.35, self.relHeights[:,0] < 0.85))[0]
        self.orientationMat = np.ones((self.torsoInds.shape[0],2),np.float)
        self.orientationMat[:,1] = self.nomWidths[self.torsoInds]

    def computeOverlaps(self, boxes):
        nA = boxes.shape[0]
        Overlaps = np.zeros((nA,nA), np.float)
        Areas = np.multiply(boxes[:,3] - boxes[:,1], boxes[:,2] - boxes[:,0])

        for i in range(nA):
            for j in range(i):
                # check in x first since that's the skinniest dimension
                x_overlap = min(boxes[i,3],boxes[j,3]) - max(boxes[i,1],boxes[j,1])
                if (x_overlap > 0):
                    x_overlap *= min(boxes[i,2],boxes[j,2]) - max(boxes[i,0],boxes[j,0])
                    if (x_overlap > 0):
                        Overlaps[i,j] = float(x_overlap) / float(Areas[i])
                        Overlaps[j,i] = float(x_overlap) / float(Areas[j])
        return Overlaps      
              

    def getBBfromPose(self, keypoint_scores, coords):
        numItems = coords.shape[0]
        BBs = np.zeros((numItems,4), np.int)
        indsToUse = self.goodInds
        A = self.MatforBB 
        for i in range(numItems):
            # first, a prelminary solution for height in pixels...                                
            Aw = (A * np.expand_dims(keypoint_scores[i,indsToUse],axis=1)).transpose()
            Ainv = np.linalg.inv(np.matmul(Aw,A))
            Sol = np.matmul(Ainv,np.matmul(Aw,np.expand_dims(coords[i,indsToUse,0],axis=1)))
            BBs[i,2] = Sol[0,0]
            BBs[i,1] = int(np.min(coords[i,:,1]))
            BBs[i,0] = Sol[0,0] - Sol[1,0]
            BBs[i,3] = int(np.max(coords[i,:,1]))
        return BBs

    def estimateOrientation(self,heightRatio, xyLengths, adjS2N,faceScores):
        # first determine whether we have a shot at representing orientation
        minS2N = 2.0
        if adjS2N < minS2N:
            # we know nothing
            return np.ones(self.numAngleBins, np.float) / self.numAngleBins

        #Angle = -np.pi / 2 *  min(1.0,heightRatio) 
        Angle = np.arcsin(min(1.0,heightRatio) )
        # 0 means angled straight forward, towards camera, 0, means pointing sideways, -pi/2 means pointing straight away from camera
        # include the twin angle as well
        angles = np.array((Angle, np.pi-Angle))
        
        rotAngle = np.arctan2(xyLengths[0],xyLengths[1])
        angles += rotAngle
        angles[angles <  -np.pi] += 2 * np.pi
        angles[angles >  np.pi] -= 2 * np.pi

        if self.angleBins == []:
            self.angleBins = np.linspace(-2*np.pi+np.pi/self.numAngleBins,2*np.pi-np.pi/self.numAngleBins,2*self.numAngleBins)
        minSigma = 0.5
        maxSigma = 2.5

        Uncertainty = 1.0 / adjS2N
        Confidence = 1.0 - 1.0 / adjS2N
        minConfidence = 1.0 - 1.0 / minS2N
        w1 = (Confidence - minConfidence) / (1 - minConfidence)

        sigma = w1 * minSigma + (1 - w1) * maxSigma
        probVec0 = np.exp(-np.square(self.angleBins-angles[0])/(2*sigma*sigma))
        probVec0 /= 2 * np.sum(probVec0)
        probVec1 = np.exp(-np.square(self.angleBins-angles[1])/(2*sigma*sigma))
        probVec1 /= 2 * np.sum(probVec1)
        probVec0 += probVec1
        probVec = probVec0[(self.numAngleBins//2):(3*self.numAngleBins//2)].copy()
        probVec[0:(self.numAngleBins//2):] += probVec0[(3*self.numAngleBins//2):(2*self.numAngleBins)]
        probVec[(self.numAngleBins//2):self.numAngleBins] += probVec0[0:(self.numAngleBins//2)]
        probVec /= np.sum(probVec)
        return probVec

    def get_person_coords_from_keypoints(self, coords, score, keypoint_scores, heightInPixels, Cam, unknownElevation = False):
        
        numPostures = self.numPostures 
        Sol2 = np.zeros((self.numPostures+1,3))
        Cov2 = np.zeros((self.numPostures+1,3,3))
        thisOrientation = np.ones(self.numAngleBins,np.float) / self.numAngleBins

        heightRatio = heightInPixels / Cam.n2

        if self.myPoseHelper.scoreTooLow(heightRatio, score):
            return False, Sol2, Cov2, np.zeros(self.numPostures,np.bool), thisOrientation, np.zeros((3,self.goodInds.size))

        if unknownElevation:
            Weights = keypoint_scores[self.goodInds]            
        else:
            Weights = np.square(keypoint_scores[self.goodInds])
        Coords = coords[self.goodInds,:]

        unitVectors = Cam.get_unit_vectors(Coords[:,::-1])

        goodInds = np.abs(unitVectors[2,:]) > 0.005
        M = np.sum(goodInds.astype(np.int))
        if M < 3:
            return False, Sol2, Cov2, np.zeros(self.numPostures,np.bool), thisOrientation, unitVectors        

        badInds = np.where(goodInds == False)[0]
        #unitVectors[2,badInds] = 1.0 # make sure we don't divide by zero
        Weights[badInds] = 0.0
        sumWeights = np.sum(Weights)

        # solve for x, y, H
        N = self.goodInds.size
     
        # we determine the xy direction from the unitvectors of the torso. 
        # Now we will pose the problem as a 2D problem (distance from camera & height)
        dirVec = np.sum(unitVectors[0:2,self.torsoInds] * Weights[self.torsoInds], axis=1) 
        dirVec /= np.linalg.norm(dirVec)
        if abs(dirVec[0]) > 0.05 and abs(dirVec[1]) > 0.05:
            
            A = self.mainMat
            b = self.mainb
            W = self.mainW
        
            uxRatios = unitVectors[0,:] / unitVectors[2,:]
            uyRatios = unitVectors[1,:] / unitVectors[2,:]
        
            A[0:N,2] = -self.relHeights[:,0] * uxRatios        
            A[N:(2*N),2] = -self.relHeights[:,0] * uyRatios
            b[0:N,0] = Cam.trans[0] - Cam.trans[2] * uxRatios
            b[N:(2*N),0] = Cam.trans[1] - Cam.trans[2] * uyRatios

            W[0:N,0] = Weights / (self.vertVars * np.square(uxRatios) + self.horVars[:,0]/2) 
            W[N:(2*N),0] = Weights / (self.vertVars * np.square(uyRatios) + self.horVars[:,0]/2) 

            # we'll regularize the height...
            # (statements below done in the init)
            #A[0:N,0] = 1.0
            #A[N:(2*N),1] = 1.0
            #A[-1,2] = 1.0
            #b[-1,0] = self.myPoseHelper.nominalPersonHeight   # height of a average person?
            #W[-1,0] = 1.0 / self.myPoseHelper.varPersonHeight # this is the uncertainty of height... +- 35 cm
            if unknownElevation:
                W[-1,0] = 0.25 / self.myPoseHelper.varPersonHeight                          # suppress the height constraint in this case, since we might rather be at a different elevation
        
            Aw = (A * W).transpose()
            Cov = np.linalg.inv(np.matmul(Aw,A))
            Sol = np.matmul(Cov,np.matmul(Aw,b))

            # new approach: we trust the xy direction of this first solution. Now we will improve
            # the weight accuracies and pose the problem as a 2D problem (distance from camera & height)
            dirVec = Sol[0:2,0] - Cam.trans[0:2]
            dirVec /= np.linalg.norm(dirVec)

        uRatios = (dirVec[0] * unitVectors[0,:] + dirVec[1] * unitVectors[1,:]) / unitVectors[2,:]
        vertVarsSqrt = np.sqrt(self.vertVars)
        minVal = 0.025
        # The following are needed when going back from the 1D to the 2D later on
        revRotMat = np.array([[dirVec[0],-dirVec[1]],[dirVec[1],dirVec[0]]])
        Trans = np.zeros((3,3),np.float)
        Trans[0:2,0:2] = revRotMat
        Trans[2,2] = 1.0

        # make it a set of 2D problems: each item represents a different pose (standing, sitting, etc)
        A2 = self.poseMat 
        b2 = self.poseb
        W2 = self.poseW

        b2[0:N,0] =  -Cam.trans[2] * uRatios
        #A2[-1,0] = 0.0
        #b2[-1,0] = self.myPoseHelper.nominalPersonHeight   # height of a average person?
        #W2[-1,0] = 1.0 / self.myPoseHelper.varPersonHeight # this is the uncertainty of height... +- 35 cm
        if unknownElevation:
            W2[-1,0] = 0.25 / self.myPoseHelper.varPersonHeight                        # suppress the height constraint in this case, since we might rather be at a different elevation
        
        Sol2a = self.Sol2a
        Cov2a = self.Cov2a
              
        bestQuickError = 1E8
        Weights2 = Weights[goodInds].copy() 
        Weights2[Coords[goodInds,0] < 0] = 0
        Weights2[Coords[goodInds,0] > Cam.n2] = 0
        Weights2[Coords[goodInds,1] < 0] = 0
        Weights2[Coords[goodInds,1] > Cam.n1] = 0
        sumWeights2 = np.sum(Weights2) 

        if sumWeights2 < 0.001:
            # sometimes pose detector returns poses completely out of frame... should look omt that, Frame 11, pose 07, in oxford dataset
            return False, Sol2, Cov2, np.zeros(self.numPostures,np.bool), thisOrientation, unitVectors   

        Error0 = np.sum(Weights2 * np.square(Coords[goodInds,0] - np.mean(Coords[goodInds,0]))) / sumWeights2
        Errors = 2 * Error0 * np.ones(self.numPostures,np.float) 
        for p in range(self.numPostures): 
            A2[0:N,1] = -self.relHeights[:,p] * uRatios
            W2[0:N,0] = Weights / (self.vertVars * np.square(uRatios) + + self.horVars[:,p]/2) 
            sumWeightsW2 = np.sum(W2)
            Aw = (A2 * W2).transpose()
            Ainv = np.linalg.inv(np.matmul(Aw,A2))
            Sol2a[p,:,:] = np.matmul(Ainv,np.matmul(Aw,b2))
            Cov2a[p,:,:] = Ainv 
            promisingPose = False

            quickError =  np.sum(W2[:,0] * np.square(np.matmul(A2,Sol2a[p,:,:]) - b2)[:]) / sumWeightsW2
            if self.elevMap.hasMap or quickError < 6.0 * bestQuickError:
                # just trying to save some compute (get_pixel_from_world_pos_new)
                promisingPose = True
                floorHeight = 0.0 
                if quickError < bestQuickError:
                    bestQuickError = quickError
                if self.myPoseHelper.postures[p] != "Lieing":
                    pos2D = Cam.trans[0:2]  +  Sol2a[p,0,0] * dirVec
                     
                    if self.elevMap.hasMap:
                        # Good place to take elevation map into account
                        thisSig = np.sqrt(Cov2a[p,0,0]) 
                        Dists, Elev = self.elevMap.getLineTransect(pos2D[0],pos2D[1],dirVec,0.25)
                        coreInds = np.where(np.abs(Dists) < 3.0*thisSig)[0]
                        if coreInds.size == 0 or np.max(np.abs(Elev[coreInds])) > 0.15:
                            # do a full 3D solution with floor height as the third unknown
                            A3 = self.poseMat3D
                            A3[0:N,2] = -self.relHeights[:,p] * uRatios #height
                            A3[0:N,1] = -uRatios                        #z
                            Aw = (A3 * W2).transpose()
                            invCov1 = np.matmul(Aw,A3)
                            Ainv = np.linalg.inv(invCov1)
                            Sol3D = np.matmul(Ainv,np.matmul(Aw,b2))
                            Cov3D = Ainv 
                            Res = np.vstack((Dists + Sol2a[p,0,0] - Sol3D[0,0],Elev - Sol3D[1,0])) 
                            zScores = np.sum(Res * np.matmul(np.linalg.inv(Cov3D[0:2,0:2]),Res), axis=0)

                            if np.min(zScores) < 6.0:
                                Wghts = np.exp(-0.5*zScores)

                                newSol = self.sol3Db
                                if 0:
                                    bestInd = np.argmax(Wghts)
                                    newSol[0:2,0] = Res[:,bestInd] + Sol3D[0:2,0]
                                else:
                                    newSol[0:2,0] = np.sum(Res * Wghts, axis=1) / np.sum(Wghts) + Sol3D[0:2,0]
                                Res += Sol3D[0:2,:] - newSol[0:2,:]

                                newCov = self.cov3D
                                newCov[0:2,0:2] = np.matmul(Res*Wghts,Res.transpose()) / np.sum(Wghts) 
                                invCov2 = np.linalg.inv(newCov)
                                Cov3 = np.linalg.inv(invCov1 + invCov2)

                                if not np.any(np.diag(Cov3) < 0.0):
                                    newSol2 = np.squeeze(np.matmul(Cov3,np.matmul(invCov1,Sol3D)) + np.matmul(Cov3,np.matmul(invCov2,newSol)))
                                    Cov2a[p,:,:] = Cov3[np.ix_(np.array((0,2)),np.array((0,2)))]
                                    Sol2a[p,:,0] = newSol2[np.array((0,2))]
                                    floorHeight = newSol2[1]
                            else:
                                # this can happen for example for a sit-on-floor-posture which will have a solution
                                # with a very wrong floor height
                                promisingPose = False

                if promisingPose:
                    pos3D = np.array([[Cam.trans[0]],[Cam.trans[1]],[0.0]])  +  Sol2a[p,0,0] * np.array([[dirVec[0]],[dirVec[1]],[0.0]]) * np.ones((1,M),np.float)
                    pos3D[2,:] += self.relHeights[goodInds,p] * Sol2a[p,1,0] + floorHeight
                    _, est_y_pixel = Cam.get_pixel_from_world_pos_new(pos3D)
                    Errors[p] = np.sum(Weights2 * np.square(est_y_pixel - Coords[goodInds,0])) / sumWeights2

        postureToInclude = Errors < min(Error0*0.35, 3.0 * np.min(Errors))
        if postureToInclude[3] and np.sum(postureToInclude) == 1:
            d = 1

        #HeightInSpec = self.myPoseHelper.heightInSpec(Sol2a[0:(-1),1,0],Cov2a[0:(-1),1,1])
        #postureToInclude = np.logical_and(postureToInclude, HeightInSpec)
        #postureToInclude = np.logical_and(HeightInSpec, Errors < 3.0 * np.min(Errors[HeightInSpec]))

        postureProbs = postureToInclude.astype(np.float) / Errors
        if np.any(postureToInclude):
            postureProbs /= np.sum(postureProbs) 

        if unknownElevation:
            normalHeight = 1.68
            normalStDev = 0.3
            Errors = 1E8 * np.ones(self.numPostures,np.float)  
            # the idea here we will provide solutions not assuming z = 0 at the foot level. We'll see how that goes
            Sol2 = np.zeros((self.numPostures,3,1))
            Cov2 = np.zeros((self.numPostures,3,3),np.float)
            Sol3 = np.zeros(4)
            Cov3 = np.zeros((4,4),np.float)

            A2 = np.zeros((N+1,3), np.float)
            b2 = np.zeros((N+1,1), np.float)
            W2 = np.zeros((N+1,1), np.float)
            A2[0:N,0] = 1.0                             # horizontal position
            A2[0:N,1] = -uRatios                        #z
            b2[0:N,0] =  -Cam.trans[2] * uRatios
            A2[-1,2] = 1.0
            b2[-1,0] = normalHeight   # height of a average person?            
            W2[-1,0] = 1.0 / np.square(normalStDev) # we enforce this height thing now in this case
            for p in range(self.numPostures): 
                if p == 3:
                    continue # lieing down lads to a degenerate matrix
                A2[0:N,2] = -self.relHeights[:,p] * uRatios #height
                W2[0:N,0] = Weights / (self.vertVars * np.square(uRatios) + + self.horVars[:,p]/2) 
                Aw = (A2 * W2).transpose()
                Ainv = np.linalg.inv(np.matmul(Aw,A2))
                Sol2[p,:,:] = np.matmul(Ainv,np.matmul(Aw,b2))
                Cov2[p,:,:] = Ainv 

                pos3D = np.array([[Cam.trans[0]],[Cam.trans[1]],[0.0]])  +  Sol2[p,0,0] * np.array([[dirVec[0]],[dirVec[1]],[0.0]]) * np.ones((1,M),np.float)
                pos3D[2,:] += self.relHeights[goodInds,p] * Sol2[p,2,0] + Sol2[p,1,0]

                _, est_y_pixel = Cam.get_pixel_from_world_pos_new(pos3D)
                Errors[p] = np.sum(Weights2 * np.square(est_y_pixel - Coords[goodInds,0])) / sumWeights2

            # transform corresponds to a rotation in the xy plane. We need to rotate back...
            # in the new coordinate system, we will say that x' is in the dirVector direction
            # so, the rotation turns [1 0] into dirVector

            # if there are multiple consistent solutions we can combine them
            kInds = np.where(Errors < 3.0 * np.min(Errors))[0]
            bestPos = kInds[0]
            if kInds.size > 1:
                return False, Sol3, Cov3, 0.0, unitVectors

            Sol3[0:2] = Cam.trans[0:2] + np.matmul(revRotMat,np.array((Sol2[bestPos,0,0], 0.0)))
            Sol3[2:4] = Sol2[bestPos,1:3,0] 
            
            Cov3[0,0] = Cov2[bestPos,0,0]
            Cov3[2,2] = Cov2[bestPos,1,1]
            Cov3[3,3] = Cov2[bestPos,2,2]
            Cov3[0,2] = Cov2[bestPos,0,1]
            Cov3[2,0] = Cov3[0,2]   
            Cov3[0,3] = Cov2[bestPos,0,2]
            Cov3[3,0] = Cov3[0,3] 
            Cov3[2,3] = Cov2[bestPos,1,2] 
            Cov3[3,2] = Cov3[2,3] 

            # add variance in the transverse direction (prior term that just describes the ill-definedness of a position (given all the body parts & pose spread out))
            Cov3[1,1] = np.mean(self.horVars[:,bestPos]/2)

            Trans2 = np.zeros((4,4))
            Trans2[0:3,0:3] = Trans
            Trans2[3,3] = 1.0

            Cov3 = np.matmul(Trans2,np.matmul(Cov3,Trans2.transpose()))
            return True, Sol3, Cov3, Sol2a[0,1,0], unitVectors

        if 0:
            weightAboveBelt = np.sum(Weights[self.relHeights[:,0] > 0.6]) / sumWeights
            weightBelowBelt = np.sum(Weights[self.relHeights[:,0] < 0.45]) / sumWeights

            with open("poseprobs.csv", "a") as csv_file:
                for p in range(self.numPostures):
                    csv_file.write(str(Errors[p])+',')
                csv_file.write(str(weightBelowBelt)+',' + str(weightAboveBelt) + ',')
                for p in range(self.numPostures):
                    csv_file.write(str(Sol2a[p,0,0])+',' + str(Sol2a[p,1,0])+ ',' + str(Cov2a[p,0,0]) + ',' + str(Cov2a[p,1,1]) + ',' + str(Cov2a[p,0,1]) + ',')
                csv_file.write('\n')
        if not np.any(postureToInclude):
            includeIt = False 
        else:
            includeIt = True           
            kInds = np.where(postureToInclude)[0]

            if kInds.size > 1:
                # first: compute aggregate version            
                Sol2a[-1,:,0] = np.sum(Sol2a[kInds,:,0] * np.expand_dims(postureProbs[kInds],axis=-1), axis = -2)
                Cov2a[-1,:,:]  = np.sum(Cov2a[kInds,:,:]  * np.expand_dims(np.expand_dims(postureProbs[kInds],axis=-1),axis=-1), axis = -3)
                Res = Sol2a[kInds,:,0] - np.expand_dims(Sol2a[-1,:,0],axis=0)
                Res2 = np.expand_dims(Res,axis=1) * np.expand_dims(Res,axis=-1)
                Cov2a[-1,:,:]  += np.sum(Res2 * np.expand_dims(np.expand_dims(postureProbs[kInds],axis=-1),axis=-1), axis = -3)

                # check for consistency (in floor position only, not in height)
                spreadRatio = np.sum((Cov2a[kInds,0,0]) * postureProbs[kInds]) / Cov2a[-1,0,0]
                if spreadRatio > 0.75:
                    # yes, they are consistent. Will merge them
                    Sol2a[kInds,:,0] = Sol2a[-1,:,0]
                    Cov2a[kInds,:,:] = Cov2a[-1,:,:]
            else:
                Sol2a[-1,:,0] = Sol2a[kInds,:,0]
                Cov2a[-1,:,:] = Cov2a[kInds,:,:]

            # transform corresponds to a rotation in the xy plane. We need to rotate back...
            # in the new coordinate system, we will say that x' is in the dirVector direction
            # so, the rotation turns [1 0] into dirVector
            
            for p in np.append(kInds,-1):
                Sol2[p,0:2] = Cam.trans[0:2] + np.matmul(revRotMat,np.array((Sol2a[p,0,0], 0.0)))
                Sol2[p,2] = Sol2a[p,1,0] 

                Cov3 = np.zeros((3,3),np.float)
                Cov3[0,0] = Cov2a[p,0,0]
                Cov3[2,2] = Cov2a[p,1,1]
                Cov3[0,2] = Cov2a[p,0,1]
                Cov3[2,0] = Cov3[0,2]   
                # add variance in the transverse direction (prior term that just describes the ill-definedness of a position (given all the body parts & pose spread out))
                Cov3[1,1] = np.mean(self.horVars[:,p]/2)
                Cov2[p,:,:] = np.matmul(Trans,np.matmul(Cov3,Trans.transpose()))
    
            if Cov2[-1,2,2] < 0.0:
                d = 1
            if not self.myPoseHelper.heightInSpec(Sol2[-1,2],Cov2[-1,2,2]):
                includeIt = False
        
        torsoInds = self.torsoInds 
        if self.doOrientation and np.sum((Weights[torsoInds] > 0.001).astype(np.int)) > 2: 
            # now try to solve for orientation...
            # solve lateral for torso area without weighting
            # make a new equation for torso
            DistFromCam = Sol2a[-1,:,0]  
            A = self.orientationMat
            bs = unitVectors[0:2,torsoInds].transpose() * DistFromCam
            Aw = (A * np.expand_dims(Weights[torsoInds],axis=1)).transpose()
            Ainv = np.linalg.inv(np.matmul(Aw, A))
            Sol4 = np.matmul(Ainv,np.matmul(Aw,bs))
            RMSE = np.mean(np.sqrt(np.sum(np.square(bs - np.matmul(A,Sol4)),axis=1)))
            latVec = Sol4[1,:]

            Magnitude = np.linalg.norm(latVec)
            heightRatio = Magnitude * Sol2[-1,2] / self.myPoseHelper.nominalPersonHeight
            AdjS2N = (self.myPoseHelper.ratioTorsoWidthToHeight * Sol2[-1,2]) / RMSE

            thisOrientation = self.estimateOrientation(heightRatio, latVec, AdjS2N, keypoint_scores[self.myPoseHelper.headInds])        
        return includeIt, Sol2, Cov2, postureProbs, thisOrientation, unitVectors        

def report_detections(myPoseSolver, camMapper, coords, scores, part_score, num):
    time0 = time.time()
    boxes = myPoseSolver.getBBfromPose(part_score, coords) 
    Overlaps = myPoseSolver.computeOverlaps(boxes)
    includeIt = myPoseSolver.myPoseHelper.detectDuplicates(scores, coords, part_score, Overlaps)

    posCov = np.zeros((num,myPoseSolver.numPoseTypes+1,3,3),np.float)
    posAndHght = np.zeros((num,myPoseSolver.numPoseTypes+1,3), np.float)
    poseTypeProb = np.zeros((num,myPoseSolver.numPoseTypes), np.float)
    Orientations = np.zeros((num,myPoseSolver.numAngleBins), np.float)
    unitVectors = np.zeros((num,3,myPoseSolver.goodInds.size), np.float)

    for i in np.where(includeIt)[0]:  
        includeIt[i], posAndHght[i,:,:], posCov[i,:,:,:], poseTypeProb[i,:], Orientations[i,:], unitVectors[i,:,:] = myPoseSolver.get_person_coords_from_keypoints(coords[i,:,:], scores[i], part_score[i,:], boxes[i,2]-boxes[i,0], camMapper)
    return includeIt, posAndHght, posCov, poseTypeProb, boxes, Orientations, time.time() - time0, unitVectors 


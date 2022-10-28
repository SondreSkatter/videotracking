import numpy as np
import bounding_box_stuff
import scipy.stats
import posenet_helper

class static_objects_pose_est:
    def __init__(self, imSize, Path, camName, usePersistedStaticObjects, poseHelper):
        self.numEls = 500   
        self.Path = Path
        self.thisCamName = camName
        self.imSize = imSize
        self.Inds = np.zeros(0,np.int)
        self.poseHelper = poseHelper
        self.numKeyPts = len(poseHelper.PART_NAMES)
        self.trueStatics = np.zeros(0,np.int)
        self.hitRatioRemovalThreshold = 0.1
        self.minNumObsTrueStatic = 8
        #self.minNumObsTrueStatic = 3000

        self.minNumOppsToRemove = 25
        
        probThres = 0.9       
        testZvals = np.linspace(0.1,2,50)
        testProbs = 1.0 - scipy.stats.chi2.cdf(testZvals,4) 
        self.zThres = testZvals[np.argmin(np.abs(testProbs-probThres))]
        self.pointVar = np.square(10.0) # the uncertainty of any one box coordinate
        self.last_frame_recieved = -1
        self.visitorWindow = 10
        self.read_persisted_static_objects(usePersistedStaticObjects)

    def save_static_objects(self):
        np.savez(self.Path+self.thisCamName+'static_objects_pose', boxes = self.boxes[self.trueStatics,:],
              points = self.points[self.trueStatics,:,:],
              points2 = self.points2[self.trueStatics,:,:], 
              pointsVar = self.pointsVar[self.trueStatics,:,:], 
              numObs = self.numObs[self.trueStatics], 
              numOpportunities = self.numOpportunities[self.trueStatics],
              numVisits = self.numVisits)
 
    def is_static_object(self, keypoints, keypointscores, indsToCheck, checkAgainstAll = False):
        # compare each of the input bounding boxes to the gallery of static boxes we've built up
        if checkAgainstAll:
            indsToCheckAgainst = self.Inds
        else:
            indsToCheckAgainst = self.trueStatics
        nA = indsToCheck.shape[0]
        matches = -np.ones(nA, np.int)

        for i in range(nA):
            Box = self.getBBfromKeypoints(keypoints[indsToCheck[i],:,:])
            for j in indsToCheckAgainst:
                if bounding_box_stuff.intersection_over_union(Box,self.boxes[j,:]) > 0.25:
                    combNorm = self.poseHelper.compareKeypoints2(keypoints[indsToCheck[i],:,:], self.points[j,:,:], keypointscores[indsToCheck[i],:], self.pointsVar[j,:,:])
                    if combNorm < 0.5:
                        matches[i] = j
                        break
        return matches

    def read_persisted_static_objects(self, usePersistedStaticObjects):
        if usePersistedStaticObjects:            
            try:
                Buff = np.load(self.Path+self.thisCamName+'static_objects_pose.npz')        
                numObj = Buff['numOpportunities'].shape[0]
                self.boxes[0:numObj,:] = Buff['boxes']
                self.points[0:numObj,:,:] = Buff['points']
                self.points2[0:numObj,:,:] = Buff['points2']
                self.pointsVar[0:numObj,:,:] = Buff['pointsVar']
                self.numObs[0:numObj] = Buff['numObs']
                self.numOpportunities[0:numObj] = Buff['numOpportunities']
                self.numVisits = Buff['numVisits']
                self.usedSlot[0:numObj] = True
                self.Inds = np.arange(0,numObj)
                self.trueStatics = self.Inds.copy()
                return
            except:
                print("No cached static objects available...")
        self.numVisits = np.zeros(self.imSize[0:2], np.float) 
        self.boxes = np.zeros((self.numEls,4), np.float)
        self.points = np.zeros((self.numEls,self.numKeyPts,2), np.float)
        self.points2 = np.zeros((self.numEls,self.numKeyPts,2), np.float)  # square
        self.pointsVar = np.zeros((self.numEls,self.numKeyPts,2), np.float)  
        self.numObs = np.zeros(self.numEls,np.float)
        self.numOpportunities = np.zeros(self.numEls,np.float)
        self.usedSlot = np.zeros(self.numEls,np.bool)

    def getStaticBoxes(self):
        return self.boxes[self.trueStatics,:]

    def updateActivityScene(self, boxes, activeInds):
        # the idea here is to, over time, identify paarts of the scene that are vnot visited and thereefore presumed to be unvisitable. It's another approach to occlusion management
        # we can trust that the bottom of the bounding box is observable
        for i in activeInds:
            self.numVisits[(boxes[i,2]-self.visitorWindow):boxes[i,2],boxes[i,1]:boxes[i,3]] += 1.0

    def likelyToBeObscured(self, box):
        # will check against the visitor map
        # if moving the box down in the image would dramatically reduce the vistor rate 
        minX = max(0,box[1])
        maxX = min(box[3]+1,self.imSize[1])
        numHorPartitions = 2
        if (maxX - minX) / numHorPartitions < 2:
            return False
        Ratios = np.zeros(numHorPartitions, np.float)
        currentAve = np.zeros(numHorPartitions, np.float)
        yRange = np.arange(max(0,box[2]-self.visitorWindow),min(self.imSize[0],box[2]+1))
        if yRange.size > 5 and maxX - minX > 0:
            currentAveVisits = np.mean(self.numVisits[yRange,minX:maxX],axis=0)
            neigborWindow = np.arange(min(box[2]+1,self.imSize[0]-1), min(box[2]+self.visitorWindow+1,self.imSize[0]))
            if neigborWindow.shape[0] > self.visitorWindow / 2:
                aveVisitsDownunder = np.mean(self.numVisits[neigborWindow,minX:maxX],axis=0)
                x0 = 0
                for k in range(numHorPartitions):
                    xInds = np.arange(x0,int((k+1)/numHorPartitions*(maxX-minX)))
                    x0 = xInds[-1]+1
                    currentAve[k] = np.sum(currentAveVisits[xInds]+0.1)
                    Ratios[k] = np.sum(aveVisitsDownunder[xInds]+0.1) / currentAve[k] 
        if np.mean(Ratios) < 0.7 or np.min(Ratios) < 0.4: 
            return True
        else:
            return False

    def getBBfromKeypoints(self, keypoints):
        box = np.zeros(4,np.float)
        box[0] = np.min(keypoints[:,0])
        box[1] = np.min(keypoints[:,1])
        box[2] = np.max(keypoints[:,0])
        box[3] = np.max(keypoints[:,1])
        return box

    def update_static_objects(self,keypoints,keypointscores,indsToCheck):
        self.last_frame_recieved += 1
        # compare each of the input bounding boxes to the gallery of static boxes we've built up
        matches = self.is_static_object(keypoints, keypointscores, indsToCheck, True)

        for i2, j in enumerate(matches):
            i = indsToCheck[i2]
            if j > -1:               
                # do the update
                self.points[j,:,:] *= self.numObs[j]
                self.points2[j,:,:] *= self.numObs[j]
                self.points[j,:,:] += keypoints[i,:,:]
                self.points2[j,:] += np.square(keypoints[i,:,:])
                self.numObs[j] += 1
                self.points[j,:,:] /= self.numObs[j] 
                self.points2[j,:,:] /= self.numObs[j] 
                self.pointsVar[j,:,:] = self.points2[j,:] - np.square(self.points[j,:]) 
            else:
                # add this one...
                # first, find a free slot where we can add it...
                for k in range(self.numEls):
                    if not self.usedSlot[k]:
                        j = k
                        break
                if j == -1:
                    # need to allocate more memory...
                    j = self.numEls
                    self.numEls = 2 * self.numEls
                    self.boxes.resize((self.numEls,4))
                    self.points.resize((self.numEls,self.numKeyPts, 2))
                    self.points2.resize((self.numEls,self.numKeyPts, 2))
                    self.pointsVar.resize((self.numEls,self.numKeyPts, 2))
                    self.numObs.resize(self.numEls)
                    self.numOpportunities.resize(self.numEls)
                    self.usedSlot.resize(self.numEls)
                self.numObs[j] = 1
                self.numOpportunities[j] = 0
                self.points[j,:,:] = keypoints[i,:,:]
                self.points2[j,:,:] = np.square(keypoints[i,:,:])
                self.pointsVar[j,:,:] = self.pointVar
                self.usedSlot[j] = True
                self.Inds = np.append(self.Inds,j)
                if np.unique(self.Inds).shape[0] < self.Inds.shape[0]:
                    d = 1
            self.boxes[j,:] = self.getBBfromKeypoints(self.points[j,:,:])
            
        # at last, see if some should be removed...
        self.numOpportunities[self.Inds] += 1
        ratioObserved = self.numObs[self.Inds] / self.numOpportunities[self.Inds]     
        
        removeThem = np.where(np.logical_and(self.numOpportunities[self.Inds]  > self.minNumOppsToRemove, ratioObserved < self.hitRatioRemovalThreshold))[0]
        if removeThem.size > 0: 
            self.usedSlot[self.Inds[removeThem]] = False
            self.Inds = np.delete(self.Inds,removeThem)
        self.trueStatics = self.Inds[self.numObs[self.Inds]  > self.minNumObsTrueStatic]

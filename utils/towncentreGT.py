import numpy as np
import re
import bounding_box_stuff
from scipy.optimize import linear_sum_assignment

# using MOTA metric: https://cvhci.anthropomatik.kit.edu/~stiefel/papers/ECCV2006WorkshopCameraReady.pdf

class towncentreGT:
    def __init__(self, GTfile, origFrameRate):
        csv = np.genfromtxt (GTfile, delimiter=",")
        self.origFrameRate = origFrameRate
        self.frame = csv[:,1].astype(np.int)
        self.personID = csv[:,0].astype(np.int)
        self.headValid = csv[:,2]
        self.bodyValid= csv[:,3]
        self.headBB = csv[:,np.array([5,4,7,6])]
        self.bodyBB = csv[:,np.array([9,8,11,10])]
        self.bodyBB[self.bodyBB[:,0]<0,0] = 0
        self.bodyBB[self.bodyBB[:,1]<0,1] = 0
        self.bodyBB[self.bodyBB[:,2]>1080,2] = 1080 
        self.bodyBB[self.bodyBB[:,3]>1920,3] = 1920 


        self.numDbIDs = int(np.max(self.personID) + 1)
        self.lastlLocIDmatch = -np.ones(self.numDbIDs, np.int)
        self.numFNs = 0
        self.numFPs = 0
        self.numMisMatches = 0
        self.numGTs = 0
        self.numFrames = 0
        self.highestPersonIDSeen = -1
        self.minOverlap = 0.25  # threshold for intersecion over union. should be 0.25-0.5 https://arxiv.org/pdf/1603.00831.pdf

    def adjustGTtoFrameRate(self, fps):
        fpsRatio = self.origFrameRate / fps
        framesToKeep = np.arange(0, self.frame.size, fpsRatio).astype(np.int)

        indsToKeep = np.where(np.in1d(self.frame, framesToKeep))[0]
        Buff, self.frame  = np.unique(self.frame[indsToKeep], return_inverse=True)
        self.personID = self.personID[indsToKeep]
        self.headValid = self.headValid[indsToKeep]
        self.bodyValid = self.bodyValid[indsToKeep]
        self.headBB = self.headBB[indsToKeep,:]
        self.bodyBB = self.bodyBB[indsToKeep,:]
        self.numDbIDs = int(np.max(self.personID) + 1)
        self.lastlLocIDmatch = -np.ones(self.numDbIDs, np.int)

    def getMOTA(self):
        return 1.0 - float(self.numFNs + self.numFPs + self.numMisMatches) / float(self.numGTs)

    def getDetRate(self):
        return 1.0 - float(self.numFNs) / float(self.numGTs)

    def getPerformanceSummary(self):
        return "MOTA is: ",self.getMOTA(), ", Detection rate: ", self.getDetRate(),", FA rate: ", self.getFARate(),", missed tracking rate: ", self.getMissedTrackings()

    def getFARate(self):
        return float(self.numFPs) / float(self.numFrames)

    def getMissedTrackings(self):
        return float(self.numMisMatches) / float(self.numGTs)

    def getBodyBB(self,frameNum):
        inds = np.where(np.logical_and(self.frame==frameNum,self.bodyValid))[0]
        return self.personID[inds], self.bodyBB[inds,:]

    def getNextBBs(self, numNextPersonsToCheck):
        # get the first occurrence of the next numNextPersonsToCheck that haven't yet been observed
        BBs = np.zeros((numNextPersonsToCheck,4), np.float)
        personIDs = np.arange(self.highestPersonIDSeen+1,self.highestPersonIDSeen+1+numNextPersonsToCheck)
        for i2, i in enumerate(personIDs):
            occurrences = np.where(np.logical_and(self.bodyValid, self.personID == i))[0]
            if occurrences.size > 0:
                BBs[i2,:] = self.bodyBB[np.min(occurrences),:]
        return personIDs, BBs

    def getClosestOccurence(self,BBID, frameNum):
        occurrences = np.where(np.logical_and(self.bodyValid, self.personID == BBID))[0]
        return self.bodyBB[occurrences[np.argmin(np.abs(occurrences-frameNum))],:]

    def check_vs_GT(self,current_boxes,current_IDs, GT_boxes, dBIDs):

        if current_boxes.ndim == 1:
            current_boxes = np.expand_dims(current_boxes,axis=0)
            current_IDs = np.expand_dims(current_IDs,axis=0)
        if GT_boxes.ndim == 1:
            GT_boxes = np.expand_dims(GT_boxes,axis=0)
            dBIDs = np.expand_dims(dBIDs,axis=0)

        nA = current_IDs.shape[0]
        nB = dBIDs.shape[0]

        isFalsePositive = np.ones(nA, np.int)
        success = np.zeros(nA,np.bool)
        isFalsePositive = np.ones(nA, np.int)
        success = np.zeros(nA,np.bool)
        numFNs = 0
        numMisMatches = 0
        
        Overlaps = np.zeros(nA, np.float)
        
        for j in range(nB):
            for i in range(nA):            
                Overlaps[i] = bounding_box_stuff.intersection_over_union(current_boxes[i], GT_boxes[j,:]) 

            bestFit = np.argmax(Overlaps)
            if Overlaps[bestFit] > self.minOverlap:
                # it was detected...
                # test if the ID is the same as earlier
                isFalsePositive[bestFit] = 0
                if (self.lastlLocIDmatch[dBIDs[j]]  == -1) or (self.lastlLocIDmatch[dBIDs[j]] == current_IDs[bestFit]):
                    # good job! detected, and same number as earlier
                    success[bestFit] = True
                else:                    
                    numMisMatches += 1
                self.lastlLocIDmatch[dBIDs[j]] = current_IDs[bestFit]
            else:
                numFNs += 1
        return success, isFalsePositive, numFNs, numMisMatches

    def check_vs_GT_Hung(self,current_boxes,current_IDs, GT_boxes, dBIDs):
        Overlaps = bounding_box_stuff.computeOverlaps(current_boxes, GT_boxes) 
        nA = Overlaps.shape[0]
        nB = Overlaps.shape[1]
        Nq = max(nA,nB)
        Obj = np.zeros((Nq,Nq))

        Obj[0:nA,0:nB] = -Overlaps

        matchesA, matchesB = linear_sum_assignment(Obj)
        non_dummy_solutions = np.where((matchesA < nA) * (matchesB < nB) * (Obj[matchesA, matchesB] < -0.25))[0]
        
        bestSum = np.sum(Obj[matchesA, matchesB])
        Diff = np.zeros(Nq)
        for qq in non_dummy_solutions:
            valCopy = Obj[matchesA[qq], matchesB[qq]] 
            Obj[matchesA[qq], matchesB[qq]] = 0.1
            resA, resB = linear_sum_assignment(Obj)
            Diff[qq] = np.sum(Obj[resA, resB]) - bestSum
            Obj[matchesA[qq], matchesB[qq]] = valCopy

        goodSolutions = non_dummy_solutions[np.where(Diff[non_dummy_solutions] > 0.2)[0]]  

        borderline_solutions = np.where((matchesA < nA) * (matchesB < nB) * (np.isin(np.arange(Nq),goodSolutions)==False) * (Obj[matchesA, matchesB] < -0.01))[0]

        #borderline_solutions = np.where((matchesA < nA) * (matchesB < nB) * (Obj[matchesA, matchesB] >= -0.25) * (Obj[matchesA, matchesB] < -0.01))[0]
        
        bordersA = matchesA[borderline_solutions]
        matchesA = matchesA[goodSolutions]
        matchesB = matchesB[goodSolutions]

        dBmatch = -np.ones(nA,np.int)
        dBmatch[matchesA] = dBIDs[matchesB]
        dBmatch[bordersA] = -0.5

        return dBmatch

    def reportResults(self, frameNum, boxesIn, annotation=[]):
        
        nA = boxesIn.shape[0]
        # very tight bounding boxes in this case... widen them a little
        boxes = boxesIn.copy() 
        addedHeightMargin = 0.18 # changed it from the old value of 0.25 when changing the BB estimation to be of top and bottom ratghe
        addedWidthMargin = 0.25

        for i in range(nA):
            Width = boxes[i,3] - boxes[i,1]
            Height = boxes[i,2] - boxes[i,0]
            boxes[i,0] = max(0,boxes[i,0] - int(addedHeightMargin * Height))
            boxes[i,1] = max(0,boxes[i,1] - int(addedWidthMargin * Width))
            boxes[i,2] = min(1080,boxes[i,2] + int(addedHeightMargin * Height))
            boxes[i,3] = min(1920,boxes[i,3] + int(addedWidthMargin * Width))

        dBIDs,dbBBs = self.getBodyBB(frameNum)
        if dBIDs.size == 0:
            return np.zeros(nA,np.bool)

        if annotation is None:
            # Only is called by the characterize_intersect_wildtrack.py
            theseInds = np.arange(nA)
            self.lastlLocIDmatch[:] = -1
            return self.check_vs_GT_Hung(boxes,theseInds, dbBBs, dBIDs)
        else:
            # first, we need to extract a uniqueID from the descriptions
            theseInds = np.zeros(nA, np.int)
            for i in range(nA):
                matchObject = re.match('(\d{1,})([a-z])',annotation[i])
                theseInds[i] = int(matchObject.group(1)) + 100 * (ord(matchObject.group(2))-97)



        if np.max(dBIDs) > self.highestPersonIDSeen:
            self.highestPersonIDSeen = np.max(dBIDs)

        nB = dbBBs.shape[0]
        self.numFrames += 1
        self.numGTs += nB

        if nA == 0:
            self.numFNs += nB
            return np.zeros(nA,np.bool)
        else:
            success, isFalsePositive, numFNs, numMisMatches = self.check_vs_GT(boxes,theseInds, dbBBs, dBIDs)
            self.numFNs += numFNs
            self.numMisMatches += numMisMatches
            
            for i in np.where(isFalsePositive == 1)[0]:
                # need to check if this is a person that has only partially entered or partially exited
                # first we'll check if it's the exit possibility
                # we can simply check if this ID was match to GT person earlier
                match_j =  np.where(self.lastlLocIDmatch == theseInds[i])[0]
                if match_j.size == 1:
                    # also check that it is not in the current view
                     if not np.any(dBIDs == match_j):
                         # need to retrieve the last bounding box for this dbID
                         thisBBbox = self.getClosestOccurence(match_j, frameNum)
                         this_success, this_isFalsePositive, buff, buff2 = self.check_vs_GT(boxes[i,:],theseInds[i], thisBBbox, match_j)
                         if this_success[0]:
                             isFalsePositive[i] = False
                             success[i] = True
                             self.numGTs += 1
                else:
                    # check if there is a person just about to enter (a few frames ahead)
                    numNextPersonsToCheck = 3
                    dBIDs,dbBBs = self.getNextBBs(numNextPersonsToCheck)
                    this_success, this_isFalsePositive, buff, buff2 = self.check_vs_GT(boxes[i,:],theseInds[i], dbBBs, dBIDs)
                    if this_success[0]:
                        isFalsePositive[i] = False
                        success[i] = True
                        self.numGTs += 1

            numFPs = np.sum(isFalsePositive)                
            self.numFPs += numFPs            
            return success



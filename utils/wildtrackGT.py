import numpy as np
import re
import bounding_box_stuff
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import graph_helper

# using MOTA metric: https://cvhci.anthropomatik.kit.edu/~stiefel/papers/ECCV2006WorkshopCameraReady.pdf
# this GT file uses world coordinates rather than image based bounding boxes 

class wildtrackGT:
    def __init__(self, GTfile, origFrameRate, sceneScope, hungarianScoring):
        csv = np.genfromtxt (GTfile, delimiter=",")
        self.SceneScope = sceneScope
        self.hungarianScoring = hungarianScoring
        if self.hungarianScoring:
            self.GTfun = self.check_vs_GT_Hungarian
        else:
            self.GTfun = self.check_vs_GT
        self.numPosDims = csv.shape[1] - 3
        self.origFrameRate = origFrameRate
        self.frame = csv[:,1].astype(np.int)
        self.personID = csv[:,0].astype(np.int)
        self.Pos = csv[:,2:(2+self.numPosDims)] 
        self.singleInd = csv[:,2+self.numPosDims].astype(np.int)

        self.numDbIDs = int(np.max(self.personID) + 1)
        self.lastlLocIDmatch = -np.ones(self.numDbIDs, np.int)
        self.numConseqMisses = np.zeros(self.numDbIDs, np.int)        
        self.numFNs = 0
        self.numFPs = 0
        self.numMisMatches = 0
        self.numGTs = 0
        self.numFrames = 0
        self.highestPersonIDSeen = -1
        self.maxDist = 1.75 #0.75  # threshold for distance (xy on the ground) 
        self.maxDist = 1.0 #2.5 #0.75  # threshold for distance (xy on the ground) 
        self.maxDist = 1.75
        # Wildtrack paper says r = 1.0

    def adjustGTtoFrameRate(self, fps):
        fpsRatio = self.origFrameRate / fps
        framesToKeep = np.arange(0, self.frame.size, fpsRatio).astype(np.int)

        indsToKeep = np.where(np.in1d(self.frame, framesToKeep))[0]
        Buff, self.frame  = np.unique(self.frame[indsToKeep], return_inverse=True)
        self.personID = self.personID[indsToKeep]
        self.Pos = self.Pos[indsToKeep,:]
        self.singleInd = self.singleInd[indsToKeep]
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

    def getBodyPos(self,frameNum):
        inds = np.where(self.frame==frameNum)[0]
        return self.personID[inds], self.Pos[inds,:]

    def getNextPos(self, numNextPersonsToCheck):
        # get the first occurrence of the next numNextPersonsToCheck that haven't yet been observed        
        personIDs = np.arange(self.highestPersonIDSeen+1,min(self.numDbIDs,self.highestPersonIDSeen+1+numNextPersonsToCheck))
        Pos = np.zeros((personIDs.size,self.numPosDims), np.float)
        for i2, i in enumerate(personIDs):
            occurrences = np.where(self.personID == i)[0]
            if occurrences.size > 0:
                Pos[i2,:] = self.Pos[np.min(occurrences),:]
        return personIDs, Pos

    def getClosestOccurence(self,BBID, frameNum):
        occurrences = np.where(self.personID == BBID)[0]
        if occurrences.size == 0:
            return 100000.0 * np.ones(2,np.float)
        frame = occurrences[np.argmin(np.abs(occurrences-frameNum))]
        return self.Pos[frame,:]

    def check_vs_GT(self,current_pos,current_cov,current_IDs, GT_poses, dBIDs):

        if current_pos.ndim == 1:
            current_pos = np.expand_dims(current_pos,axis=0)
            current_cov = np.expand_dims(current_cov,axis=0)
            current_IDs = np.expand_dims(current_IDs,axis=0)
        if GT_poses.ndim == 1:
            GT_poses = np.expand_dims(GT_poses,axis=0)
            dBIDs = np.expand_dims(dBIDs,axis=0)

        nA = current_IDs.shape[0]
        nB = dBIDs.shape[0]

        isFalsePositive = np.ones(nA, np.int)
        isFalseNegative = np.ones(nB, np.int)
        
        success = np.zeros(nA,np.bool)
        numFNs = 0
        numMisMatches = 0
        
        interDists = np.zeros((nA,nB), np.float)
        interDistsZ = 1000 * np.ones((nA,nB), np.float)
        likelihood = np.zeros((nA,nB), np.float)
        Vars = np.zeros(nA, np.float)
        
        for i in range(nA):   
            Vars[i] = np.trace(current_cov[i,:,:])
            Cov = current_cov[i,:,:].copy() + np.square(0.3) * np.eye(2)
            Const = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(Cov)))
            for j in range(nB):
                Diff = np.expand_dims(current_pos[i,0:2] - GT_poses[j,0:2],axis=0)                               
                interDists[i,j] = np.linalg.norm(Diff[0,:]) 
                interDistsZ[i,j] =  np.matmul(Diff,np.matmul(np.linalg.inv(Cov),Diff.transpose())) 
                likelihood[i,j] = Const * np.exp(-0.5 * interDistsZ[i,j])
        interDistsCopy = interDists.copy()
        while nB > 0:
            i, j = np.unravel_index(np.argmin(interDists.ravel()),(nA,nB))
            #i, j = np.unravel_index(np.argmax(likelihood.ravel()),(nA,nB))
            if interDists[i, j] < self.maxDist:
                # it was detected...
                # test if the ID is the same as earlier
                isFalsePositive[i] = 0
                isFalseNegative[j] = 0
                if (self.lastlLocIDmatch[dBIDs[j]]  == -1) or (self.lastlLocIDmatch[dBIDs[j]] == current_IDs[i]) or (self.numConseqMisses[dBIDs[j]] > 10):
                    # good job! detected, and same number as earlier
                    success[i] = True
                else:                    
                    numMisMatches += 1
                self.lastlLocIDmatch[dBIDs[j]] = current_IDs[i]
                self.numConseqMisses[dBIDs[j]] = 0
                # make sure neither i nor j gets picked again
                interDists[i,:] = 10000.0
                interDists[:,j] = 10000.0
                #interDistsZ[i,:] = 10000.0
                #interDistsZ[:,j] = 10000.0
                #likelihood[i,:] = 0.0
                #likelihood[:,j] = 0.0
            else:
                break

        numFNs = np.sum(isFalseNegative)
        return success, isFalsePositive, numFNs, numMisMatches, dBIDs[isFalseNegative==1], interDistsCopy

    def check_vs_GT_Hungarian(self,current_pos,current_cov,current_IDs, GT_poses, dBIDs):

        if current_pos.ndim == 1:
            current_pos = np.expand_dims(current_pos,axis=0)
            current_cov = np.expand_dims(current_cov,axis=0)
            current_IDs = np.expand_dims(current_IDs,axis=0)
        if GT_poses.ndim == 1:
            GT_poses = np.expand_dims(GT_poses,axis=0)
            dBIDs = np.expand_dims(dBIDs,axis=0)

        nA = current_IDs.shape[0]
        nB = dBIDs.shape[0]

        isFalsePositive = np.ones(nA, np.int)
        isFalseNegative = np.ones(nB, np.int)
        success = np.zeros(nA,np.bool)
        numFNs = 0
        numMisMatches = 0
        
        interDists = np.zeros((nA,nB), np.float)
        interDistsZ = 1000 * np.ones((nA,nB), np.float)
        likelihood = np.zeros((nA,nB), np.float)
        Vars = np.zeros(nA, np.float)
        
        for i in range(nA):   
            Vars[i] = np.trace(current_cov[i,:,:])
            Cov = current_cov[i,:,:].copy() + np.square(0.3) * np.eye(2)
            Const = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(Cov)))
            for j in range(nB):
                Diff = np.expand_dims(current_pos[i,0:2] - GT_poses[j,0:2],axis=0)                               
                interDists[i,j] = np.linalg.norm(Diff[0,:]) 
                interDistsZ[i,j] =  np.matmul(Diff,np.matmul(np.linalg.inv(Cov),Diff.transpose())) 
                likelihood[i,j] = Const * np.exp(-0.5 * interDistsZ[i,j])

        DistsReturned = interDists.copy()
        NSq = max(likelihood.shape[0], likelihood.shape[1])
        if 0:
            zeroProb = 1E-10                        
            likelihood[likelihood < zeroProb] = zeroProb
            Weights = -np.log(zeroProb) * np.ones((NSq,NSq),np.float)
            Weights[0:likelihood.shape[0],0:likelihood.shape[1]] = -np.log(likelihood) 
        else:
            Weights = 10.0 * np.ones((NSq,NSq),np.float)
            Weights[0:likelihood.shape[0],0:likelihood.shape[1]] = interDists
        #matchesA, matchesB = linear_sum_assignment(Weights)
        if 0:
            Weights[Weights>10] = 10
            matchesA, matchesB = linear_sum_assignment(-np.log(np.exp(-Weights*Weights/2)))
            non_dummy_solutions = np.where(Weights[matchesA, matchesB] < self.maxDist)[0]
        else:
            matchesA, matchesB = graph_helper.hungarianDists(interDists, self.maxDist)
            non_dummy_solutions = np.arange(matchesA.size)

        

        if 0:
            bestSum = np.sum(Weights[matchesA, matchesB])
            Diff = np.zeros(NSq)
            for qq in non_dummy_solutions:
                valCopy = Weights[matchesA[qq], matchesB[qq]] 
                Weights[matchesA[qq], matchesB[qq]] = 10000.0 #-np.log(zeroProb)
                resA, resB = linear_sum_assignment(Weights)
                Diff[qq] = np.log(np.sum(Weights[resA, resB])) - np.log(bestSum)
                Weights[matchesA[qq], matchesB[qq]] = valCopy

        goodSolutions = non_dummy_solutions #[np.where(Diff[non_dummy_solutions] > np.log(1.1))[0]]  

        interDists2 = np.ones((nA,nB)) * 1000.0
        interDists2[matchesA[goodSolutions],matchesB[goodSolutions]] = interDists[matchesA[goodSolutions],matchesB[goodSolutions]] 

        while nB > 0:
            i, j = np.unravel_index(np.argmin(interDists2.ravel()),(nA,nB))
            if interDists2[i, j] < self.maxDist:
                # it was detected...
                # test if the ID is the same as earlier
                isFalsePositive[i] = 0
                isFalseNegative[j] = 0
                if (self.lastlLocIDmatch[dBIDs[j]]  == -1) or (self.lastlLocIDmatch[dBIDs[j]] == current_IDs[i]) or (self.numConseqMisses[dBIDs[j]] > 10):
                    # good job! detected, and same number as earlier
                    success[i] = True
                else:                    
                    numMisMatches += 1
                self.lastlLocIDmatch[dBIDs[j]] = current_IDs[i]
                self.numConseqMisses[dBIDs[j]] = 0
                # make sure neither i nor j gets picked again
                interDists2[i,:] = 10000.0
                interDists2[:,j] = 10000.0
            else:
                break

        numFNs = np.sum(isFalseNegative)
        return success, isFalsePositive, numFNs, numMisMatches, dBIDs[isFalseNegative==1], DistsReturned


    def reportResults(self, frameNum, posInIn, posCovIn, ID, numReincarnations):       
        # we will disregard folks outside of the sub-area in where people are matched
        goodInds = np.logical_and(np.logical_and(posInIn[:,0] > self.SceneScope[1], posInIn[:,0] < self.SceneScope[3]),
            np.logical_and(posInIn[:,1] > self.SceneScope[0], posInIn[:,1] < self.SceneScope[2]))

        NA = posInIn.shape[0]
        posIn = posInIn[goodInds,:].copy()
        posCov = posCovIn[goodInds,:,:].copy()
        nA = np.sum(goodInds.astype(np.int))
        revInd = -np.ones(NA,np.int)
        revInd[goodInds] = np.arange(nA)

        # first, we need to extract a uniqueID from the descriptions
        theseInds = ID[goodInds] + 1000 * numReincarnations[goodInds]

        dBIDsAll,dbPosAll = self.getBodyPos(frameNum)


        presentThisFrame = np.zeros(self.numDbIDs, np.bool)
        presentThisFrame[dBIDsAll] = True
        missedThisTime = np.where(presentThisFrame == False)[0]
        self.numConseqMisses[missedThisTime] += 1


        revIndB = np.zeros(np.max(dBIDsAll)+1,np.int)
        revIndB[dBIDsAll] = np.arange(dBIDsAll.size)
        if dBIDsAll.size == 0:
            return np.zeros(nA,np.bool), dBIDsAll, ['Nuisance alarm'] * NA, [], dbPosAll

        if np.max(dBIDsAll) > self.highestPersonIDSeen:
            self.highestPersonIDSeen = np.max(dBIDsAll)

        nB = dbPosAll.shape[0]
        self.numFrames += 1
        self.numGTs += nB

        retVal = np.zeros(NA,np.int)
        if nA == 0:
            self.numFNs += nB
            return retVal, dBIDsAll, [], ['Miss'] * nB, dbPosAll
        else:            
            success, isFalsePositive, numFNs, numMisMatches, falseNegs, Dists = self.GTfun(posIn,posCov,theseInds, dbPosAll, dBIDsAll)
            self.numFNs += numFNs
            self.numMisMatches += numMisMatches
            
            for i in np.where(isFalsePositive == 1)[0]:
                # need to check if this is a person that has only partially entered or partially exited
                # first we'll check if it's the exit possibility
                # we can simply check if this ID was match to GT person earlier
                match_j =  np.where(self.lastlLocIDmatch == theseInds[i])[0]
                if match_j.size == 1:
                    # also check that it is not in the current view
                     if not np.any(dBIDsAll == match_j):
                         # need to retrieve the last bounding box for this dbID
                         thisPos = self.getClosestOccurence(match_j, frameNum)
                         this_success, this_isFalsePositive, buff, buff2, _, _ = self.GTfun(posIn[i,:],posCov[i,:,:],theseInds[i], thisPos, match_j)
                         if this_success[0]:
                             isFalsePositive[i] = 0
                             success[i] = True
                             self.numGTs += 1
                else:
                    # check if there is a person just about to enter (a few frames ahead)
                    numNextPersonsToCheck = 3
                    dBIDs,dbPos = self.getNextPos(numNextPersonsToCheck)
                    if dBIDs.size > 0:
                        this_success, this_isFalsePositive, buff, buff2, _, _ = self.GTfun(posIn[i,:],posCov[i,:,:],theseInds[i], dbPos, dBIDs)
                        if this_success[0]:
                            isFalsePositive[i] = 0
                            success[i] = True
                            self.numGTs += 1

            numFPs = np.sum(isFalsePositive)                
            self.numFPs += numFPs    
            retVal[goodInds] = 2 * (success.astype(np.int) - 0.5) - isFalsePositive.astype(np.int)  # 0: not scored, 1: correct det, -2: false positive, -1: mistracked

            if numFPs > 0 or np.any(retVal < 0) or falseNegs.size > 0:
                d = 1
            # Will try to assess failure modes also, i.e. what went wrong when it went wrong...
            # First the positive observations
            posMisses = np.where(retVal < 0)[0]
            positiveCodes = [None] * NA
            distTolerance = self.maxDist * 2
            for i2, i in enumerate(posMisses):    
                i3 = revInd[i]
                if retVal[i] == -2:
                    # False positives
                    bestMatch = np.argmin(Dists[i3,:])
                    if Dists[i3,bestMatch] < distTolerance:
                        if bestMatch in falseNegs:
                            positiveCodes[i] = 'Wrong position'
                        else:
                            positiveCodes[i] = 'Duplicate'
                    else:
                        positiveCodes[i] = 'Nuisance alarm'
                else:
                    # Mis-tracked
                    # Two failure modes: Swap, or duplicate
                    # look for an accompanying false positive close by
                    peerDists = np.linalg.norm(posIn[np.where(isFalsePositive)[0],0:2] - posIn[i3,0:2],axis=-1)
                    if np.any(peerDists < distTolerance):
                        positiveCodes[i] = 'Duplicate'
                    else:
                        positiveCodes[i] = 'ID swap'
                   
            negativeCodes = [None] * falseNegs.size
            for j2, j in enumerate(falseNegs):
                # false negatives
                if np.any(Dists[revInd[np.where(isFalsePositive==1)[0]],revIndB[j]] < distTolerance):
                    negativeCodes[j2] = 'Wrong position'
                else:
                    negativeCodes[j2] = 'Miss'                

            # Return also posiions of the true Negatives
            negPoses = np.zeros((falseNegs.size,self.numPosDims))
            for i2, i in enumerate(falseNegs):
                j = np.where(dBIDsAll == i)[0]
                negPoses[i2,:] = dbPosAll[j,:]

            return retVal, falseNegs, positiveCodes, negativeCodes, negPoses  #retVal:  0: not scored, 1: correct det, -2: false positive, -1: mistracked

    def hungar(self,posA,posB):
        Nq = max(posA.shape[0],posB.shape[0])
        Dists = 1000.0 * np.ones((Nq,Nq))

        Dists[0:posA.shape[0],0:posB.shape[0]] = np.sqrt(np.square(np.expand_dims(posA[:,0],axis=1) - posB[:,0]) + np.square(np.expand_dims(posA[:,1],axis=1) - posB[:,1]))

        matchesA, matchesB = linear_sum_assignment(Dists)
        #non_dummy_solutions = np.where(np.logical_and(matchesA < posA.shape[0], matchesB < posB.shape[0]))[0]
        non_dummy_solutions = np.where(Dists[matchesA,matchesB] < self.maxDist)[0]

        matchesA = matchesA[non_dummy_solutions]
        matchesB = matchesB[non_dummy_solutions]
        return matchesA, matchesB

    def hungar2(self,DistsIn):
        Nq = max(DistsIn.shape[0],DistsIn.shape[1])
        Dists = 1000.0 * np.ones((Nq,Nq))

        Dists[0:DistsIn.shape[0],0:DistsIn.shape[1]] = DistsIn

        matchesA, matchesB = linear_sum_assignment(Dists)
        non_dummy_solutions = np.where(np.logical_and(matchesA < DistsIn.shape[0], matchesB < DistsIn.shape[1]))[0]
        matchesA = matchesA[non_dummy_solutions]
        matchesB = matchesB[non_dummy_solutions]
        return matchesA, matchesB


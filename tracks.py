import numpy as np
import copy

class track:
    def __init__(self):
        self.allocBlockSize = 100
        self.numAlloc = 200
        self.numObs = 0
        self.Pos = np.zeros((self.numAlloc,2))
        self.Time = np.zeros(self.numAlloc)
        self.IDref = np.zeros(2,np.int)
        self.timeScope = np.zeros(2) # min max of time posx, posy, and time
        self.posScope = np.zeros((2,2)) # min max of time posx, posy, and time

        self.colorFeats = None
        self.colorCov = None

    def clear(self):
        self.numObs = 0

    def newTrack(self, ID, reincarn, Pos, Time):
        self.numObs = 0
        self.IDref = np.array((ID,reincarn))
        self.addObservation(Pos, Time)
        self.timeScope[0] = Time
        self.posScope[:,:] = np.expand_dims(Pos,axis=0)

    def setColor(self, colorFeats, colorCov):
        self.colorFeats = colorFeats
        self.colorCov = colorCov

    def addObservation(self, Pos, Time):
        if self.numObs > self.numAlloc - 2:
            self.reallocTracks()
        self.Pos[self.numObs,:] = Pos
        self.Time[self.numObs] = Time
        self.numObs += 1
        self.timeScope[1] = Time
        self.posScope[0,0] = min(self.posScope[0,0],Pos[0])
        self.posScope[0,1] = max(self.posScope[0,1],Pos[0])
        self.posScope[1,0] = min(self.posScope[1,0],Pos[1])
        self.posScope[1,1] = max(self.posScope[1,1],Pos[1])

    def getDist(self,Pos,Time,maxDist):
        if Time > self.timeScope[0] - maxDist[1] and Time < self.timeScope[1] + maxDist[1]:
            # fits in the time window...
            if Pos[0] > self.posScope[0,0] - maxDist[0] and Pos[0] < self.posScope[0,1] + maxDist[0] and \
                Pos[1] > self.posScope[1,0] - maxDist[0] and Pos[1] < self.posScope[1,1] + maxDist[0]:
                # find all the observations within the time window...
                Inds = np.where(np.abs(self.Time[0:self.numObs] - Time) < maxDist[1])[0]
                Dists = np.linalg.norm(self.Pos[Inds,:] - Pos, axis = -1)
                bestInd = np.argmin(Dists)
                if Dists[bestInd] < maxDist[0]:
                    return np.array((Dists[bestInd],self.Time[Inds[bestInd]] - Time))
        return None

    def reallocTracks(self):
        self.numAlloc += self.allocBlockSize
        self.Pos.resize((self.numAlloc,2))
        self.Time.resize(self.numAlloc)


# mostly just a data object to hold the tracks for a set of persons in a scene

class tracks:
    def __init__(self, numChannels, myChiDistGuy, numColorFeats, postureMobile, numAlloc, numAngleBins, snapshotOnly, doFaceID, unknownHeight=False):   
        self.myChiDistGuy = myChiDistGuy
        self.numChannels = numChannels
        self.numColorFeats = numColorFeats 
        self.postureMobile = postureMobile
        self.numPoseTypes = postureMobile.shape[0]
        self.framesProcessed = np.zeros(numChannels,np.int)
        self.snapshotOnly = snapshotOnly
        self.numAlloc = numAlloc
        self.allocBlockSize = 50
        self.maxNumIDs = 10000
        maxReasonableHumanSpeedMinPerMile = 5.5 # 5:30 minutes per mile running pace. It's fast
        maxReasonableHumanSpeedMetersPerSec = 1609 / (maxReasonableHumanSpeedMinPerMile * 60)

        self.veloVar = np.square(maxReasonableHumanSpeedMetersPerSec / 2)  # Setting sigma to half of this max speed, essentially putting the max speed @ 2 sigma
        self.maxHumanAccel = 2.0 # 2.0 m / sec2 
        self.humanShortTermShift = 0.3 # shift of center of gravity in a short term, shuffley walk etc

        N = self.numAlloc
        self.Inds = np.zeros(0,np.int) # using an indexing mechanism so we don't have to move memory around when removing a human
        self.lastSeenIDs = np.zeros(0,np.int) 
        self.slotUsed = np.zeros(N, np.bool)
        self.doFaceID = doFaceID

        self.numObs = np.zeros((N,self.numChannels), np.int)
        if unknownHeight:
            self.numPosDims = 4
        else:
            self.numPosDims = 3
        self.posAndHeight = np.zeros((N,self.numPoseTypes,self.numPosDims), np.float)  # position in meters
        self.posCov = np.zeros((N,self.numPoseTypes,self.numPosDims,self.numPosDims), np.float) 
        self.poseTypeProb = np.zeros((N,self.numPoseTypes), np.float) 

        self.numAngleBins = numAngleBins
        self.bodyOrientation = np.ones((N,self.numAngleBins),np.float) / self.numAngleBins
        self.colorFeats = np.zeros((N,self.numChannels,self.numColorFeats), np.float)
        self.colorCov = np.zeros((N,self.numChannels,self.numColorFeats,self.numColorFeats), np.float)
        self.nmFrmsHrznColor = 5
        self.nmFrmsHrznColor = 10
        self.numColorObs = np.zeros((N,self.numChannels), np.int)  
        self.numReincarnations = np.zeros(N, np.int)
        self.temp_ids = [None] * N

        if snapshotOnly:
            self.Score = np.zeros(N, np.float)
        else:
            self.Track = []
            for ii in range(N):
                self.Track.append(track())
            self.archivedTracks = []
            self.poseTypeProbPrev = np.zeros((N,self.numPoseTypes), np.float) 
            self.poseTypeProbCurrent = np.zeros((N,self.numPoseTypes), np.float)             
            self.poseTypeProbVelo = np.zeros((N,self.numPoseTypes), np.float)
            self.lastTimeStamp = 0.0
            self.probDuplicate = np.zeros((N,N), np.float) # an indicator that a pair of persons are teh same person, i.e. duplicate representations
            self.nConseqMisses = np.zeros((N,self.numChannels), np.int)
            self.nConseqHits = np.zeros((N,self.numChannels), np.int)
            self.lastTimeStampPerPerson = np.zeros(N, np.float)
            self.posAndHeightTemp = np.zeros((N,self.numPoseTypes+1,self.numPosDims), np.float)  # position in meters        
            self.posCovTemp = np.zeros((N,self.numPoseTypes+1,self.numPosDims,self.numPosDims), np.float)

            # Kalman filter data items
            # Wikipedia or https://github.com/skhobahi/Kalman-Filter-Object-Tracking
            self.F0 = np.eye(5)
            # dT dependent portion
            self.FdT = np.zeros((5,5))
            self.FdT[0,3] = 1.0
            self.FdT[1,4] = 1.0
            self.H = np.zeros((3,5))
            self.H[0,0] = 1.0
            self.H[1,1] = 1.0
            self.H[2,2] = 1.0
            self.Q0 = np.diag(np.array((np.square(self.humanShortTermShift),np.square(self.humanShortTermShift),0.0,0.0,0.0)))
            self.dTcutoff = 0.5 # limit the divergence of the covariance matrix under lack of observation
            self.processSigma2 = 2*0.5 * 0.25 * np.square(self.maxHumanAccel)
            self.KalmanState = np.zeros((N,self.numPoseTypes,5), np.float)
            self.KalmanCov = np.zeros((N,self.numPoseTypes,5,5), np.float)
            self.KalmanStateUpdate = np.zeros((N,self.numPoseTypes,5), np.float)
            self.KalmanCovUpdate = np.zeros((N,self.numPoseTypes,5,5), np.float)

            if doFaceID:
                self.db_ids = [None] * N            
                self.names = [''] * N
                self.face_images = [None] * N
                self.face_detected = np.zeros(N, np.bool)
 
    def frameProcessed(self,channelNum):
        if channelNum < self.numChannels:
            self.framesProcessed[channelNum] += 1

    def clear(self):
        assert self.snapshotOnly, "This function intended only for snapshotOnly.."
        self.slotUsed[:] = False 
        self.numObs[self.Inds,:] = 0
        self.Inds.resize(0)

    def removeHuman(self, i):
        self.slotUsed[i] = False
        self.numReincarnations[i] += 1
        self.nConseqMisses[i,:] = 0
        self.nConseqHits[i,:] = 0
        ind = np.where(self.Inds == i)[0]
        self.Inds = np.delete(self.Inds, ind)   
        self.probDuplicate[i,:] = 0
        self.probDuplicate[:,i] = 0
        if not self.snapshotOnly and np.sum(self.numObs[i,:]) > 10:
            self.Track[i].setColor(self.colorFeats[i,:,:], self.colorCov[i,:,:,:])
            self.archivedTracks.append(copy.deepcopy(self.Track[i]))        

    def uniqishStringID(self, i):
        return str(i) + str(chr(97+(self.numReincarnations[i] % 26)))

    def makeCaption(self, i):
        Position = ""
        if self.poseTypeProb[i,0] < 0.85:
            if np.sum(self.poseTypeProb[i,1:3]) > 0.55:
                Position = "sitting. "

        FaceCap = ""
        if self.doFaceID:
            if self.face_detected[i]:
                FaceCap = ": face detected"    
                if self.db_ids[i]:
                    FaceCap = ": " + self.names[i] if self.names[i] != '' else self.db_ids[i]
        return self.temp_ids[i] + Position + FaceCap
    
    def getUniqueID(self,IDs): return IDs + self.maxNumIDs * np.mod(self.numReincarnations[IDs], self.maxNumIDs)

    def getIDs(self, uniqueIDS):
        # utility function to evaluate a of uniquified IDs from a child channel tracker node
        IDpart = np.mod(uniqueIDS, self.maxNumIDs)
        reincarPart = (uniqueIDS / self.maxNumIDs).astype(np.int)
        goodOnes = np.where(reincarPart - self.numReincarnations[IDpart] == 0)[0]
        return IDpart[goodOnes], goodOnes

    def mergePeople(self, j1, j2):
        # merge all the data from two persons. Decided which entry to keep. Return the ID for the keeper
        Score1 = np.sum(self.numObs[j1,:])
        Score2 = np.sum(self.numObs[j2,:])
        if self.doFaceID:
            if self.db_ids[j1]:  
                Score1 += 100
            if self.db_ids[j2]:  
                Score2 += 100

        if Score1 > Score2:
            keeper = j1
            loser = j2
        else:
            keeper = j2
            loser= j1
        # do the color stuff
        for c in range(self.numChannels):
             # blend new color stats with c
            numSamples1 = min(self.nmFrmsHrznColor,self.numColorObs[j1,c])
            numSamples2 = min(self.nmFrmsHrznColor,self.numColorObs[j2,c])
            if numSamples1 + numSamples2 > 0:
                w1 = numSamples1 / (numSamples1 + numSamples2)
                w2 = 1.0 - w1

                oldVals1  = self.colorFeats[j1,c,:].copy()
                oldVals2  = self.colorFeats[j2,c,:].copy()
                self.colorFeats[keeper,c,:] = w1 * self.colorFeats[j1,c,:] + w2 * self.colorFeats[j2,c,:]
                self.colorCov[keeper,c,:,:] = w1 * self.colorCov[j1,c,:,:] + w2 * self.colorCov[j2,c,:,:]
                self.colorCov[keeper,c,:,:] += w1 * np.multiply(oldVals1-self.colorFeats[j1,c,:] ,np.expand_dims(oldVals1-self.colorFeats[j1,c,:] ,axis=1)) + w2 * np.multiply(self.colorFeats[j2,c,:] -oldVals2,np.expand_dims(self.colorFeats[j2,c,:] -oldVals2,axis=1)) 
    
        # then position and height
        if np.sum(self.nConseqHits[j1,:]) > 0 and np.sum(self.nConseqHits[j2,:]) == 0: 
            posKeeper = j1
        elif np.sum(self.nConseqHits[j2,:]) > 0 and np.sum(self.nConseqHits[j1,:]) == 0:
            posKeeper = j2
        else:
            posKeeper = keeper

        self.posAndHeight[keeper,:,:] = self.posAndHeight[posKeeper,:,:]
        self.posCov[keeper,:,:,:] = self.posCov[posKeeper,:,:,:]
        self.lastTimeStampPerPerson[keeper] = self.lastTimeStampPerPerson[posKeeper] 
        self.poseTypeProb[keeper,:] = self.poseTypeProb[posKeeper,:] 
        self.poseTypeProbPrev[keeper,:] = self.poseTypeProbPrev[posKeeper,:] 
        self.poseTypeProbCurrent[keeper,:] = self.poseTypeProbCurrent[posKeeper,:] 

        self.KalmanState[keeper,:] = self.KalmanState[posKeeper,:] 
        self.KalmanCov[keeper,:,:] = self.KalmanCov[posKeeper,:,:] 
        self.KalmanStateUpdate[keeper,:] = self.KalmanStateUpdate[posKeeper,:] 
        self.KalmanCovUpdate[keeper,:,:] = self.KalmanCovUpdate[posKeeper,:,:] 

        self.numObs[keeper, :] += self.numObs[loser, :]   
        self.nConseqMisses[keeper,:] = np.min(np.array((self.nConseqMisses[keeper,:],self.nConseqMisses[loser,:])),axis=0) 
        self.nConseqHits[keeper,:] = self.nConseqMisses[keeper,:] + self.nConseqMisses[loser,:]
        
        self.removeHuman(loser)
        return keeper  

    def whoWasThere(self, Pos, Time):
        # look through current and archived tracks for matching time and place
        IDs = np.zeros((0,2),np.int)
        Dists = np.zeros((0,2)) # spatial & temporal distance
        distMargin = 1.0 # if two person are within closer than this we'll have to return both
        maxDist = np.array((2.5,0.5))
        for i in self.Inds:
            thisDist = self.Track[i].getDist(Pos,Time,maxDist)
            if not thisDist is None:
                Dists = np.vstack((Dists,thisDist))
                IDs = np.vstack((IDs,self.Track[i].IDref))
        for i in range(len(self.archivedTracks)):
            thisDist = self.archivedTracks[i].getDist(Pos,Time,maxDist)
            if not thisDist is None:
                Dists = np.vstack((Dists,thisDist))
                IDs = np.vstack((IDs,self.archivedTracks[i].IDref))

        if Dists.shape[0] > 0:
            if Dists.shape[0] == 1:
                Inds = 0
            else:
                Inds = Dists[:,0].argsort()[0:2] # the two smallest ones
                if Dists[Inds[1],0] - Dists[Inds[0],0] > distMargin:
                    Inds = Inds[0]
        else:
            return IDs        
        return IDs[Inds,:]

    def predict_kalman(self, timeStamp):
        # We are further ahead in time from the last observation and will make temporary estimates of positions by 
        # utilizing the velocity and the elapsed time
        for j in self.Inds:
            dT = timeStamp - self.lastTimeStampPerPerson[j]            
            dTcapped  = min(dT,self.dTcutoff)
            dTfactor = dT / dTcapped # Idea being: a human can only accelerate for so long. after that it becomes random walk
            for k in np.where(self.poseTypeProb[j,:] > 0.01)[0]:
                if self.postureMobile[k]:
                    F = self.F0 + dT * self.FdT 
                    Q = self.Q0 +  self.processSigma2 * np.array(((dTcapped**4/4, 0, 0, dTcapped**3/2,0),(0,dTcapped**4/4,0,0,dTcapped**3/2),
                        (0,0,0,0,0),(dTcapped**3/2,0,0,dTcapped**2,0),(0,dTcapped**3/2,0,0,dTcapped**2)))*dTfactor
                else:
                    F = self.F0 
                    Q = self.Q0 * 0
                self.KalmanStateUpdate[j,k,:] = np.matmul(F, self.KalmanState[j,k,:])
                self.KalmanCovUpdate[j,k,:,:] = np.matmul(np.matmul(F, self.KalmanCov[j,k,:,:]), F.transpose()) + Q

                # Update legacy data (for now)
                self.posAndHeightTemp[j,k,:] = self.KalmanStateUpdate[j,k,0:3]
                self.posCovTemp[j,k,:,:] = self.KalmanCovUpdate[j,k,0:3,0:3]

            # lastly, vagueify the orientation as well
            self.bodyOrientation[j,:] += 0.25 / self.numAngleBins
            self.bodyOrientation[j,:] /= np.sum(self.bodyOrientation[j,:])

        self.computeAggregatePosInternal(self.Inds)
        self.lastTimeStamp = timeStamp

    def colorFeatsValid(self,Inds):
        return self.numColorObs[Inds,:] > 0

    def reportOfMisses(self,channelNum,Inds):
        if channelNum == -1:
            self.nConseqMisses[Inds,:] += 1
            self.nConseqHits[Inds,:] = 0
        else:
            self.nConseqMisses[Inds,channelNum] += 1
            self.nConseqHits[Inds,channelNum] = 0

    def computeAggregatePosInternal(self, Inds):

        if self.snapshotOnly:
            posHeight = self.posAndHeight
            posCov = self.posCov
        else:
            posHeight = self.posAndHeightTemp
            posCov = self.posCovTemp


        for i in Inds:
            kInds = np.where(self.poseTypeProb[i,:] > 0.01)[0]

            assert kInds.size > 0, "No non-zeros probs. better look into this one..."

            posHeight[i,-1,:] = np.sum(posHeight[i,kInds,:] * np.expand_dims(self.poseTypeProb[i,kInds],axis=-1), axis = -2)
            posCov[i,-1,:,:]  = np.sum(posCov[i,kInds,:,:]  * np.expand_dims(np.expand_dims(self.poseTypeProb[i,kInds],axis=-1),axis=-1), axis = -3)
            if kInds.size > 1:
                Res = posHeight[i,kInds,:] - np.expand_dims(posHeight[i,-1,:],axis=0)
                Res2 = np.expand_dims(Res,axis=1) * np.expand_dims(Res,axis=-1)
                posCov[i,-1,:,:]  += np.sum(Res2 * np.expand_dims(np.expand_dims(self.poseTypeProb[i,kInds],axis=-1),axis=-1), axis = -3)

    def posesAreConsistent(self, Pos, Cov, Prob):
        # if there are multiple poses possible, gage whether they are: 
        # a) consistent, which implies and intermediary state, or
        # b) inconsistent, which implies different mutually exclusive posititions (happens due to occlusions)
        Types = np.where(Prob > 0.01)[0]
        if Types.size < 2:
            return True
        spreadRatio = np.sum((Cov[Types,0,0] + Cov[Types,1,1]) * Prob[Types]) / (Cov[-1,0,0] + Cov[-1,1,1])
        return spreadRatio > 0.75

    def reconcileProbabilities(self, posNew, covNew, probsNew, posOld, covOld, probsOld, probsCurrent,scoreNew=[],scoreOld=[]):
        # first: if there are more than one common category, analyze the joint probabilities
        outProbs = np.zeros(self.numPoseTypes,np.float)
        Types = np.where(probsNew + probsOld > 0.01)[0]
        if Types.size == 1:
            return probsOld # assume probsNew is normalized correctly...

        commonTypes = np.where(probsNew * probsOld > 0.0001)[0]

        if commonTypes.size == 0:
            # this means that the comparison came through favorably by using two different categories. Fine
            kOld = np.argmax(probsOld)
            kNew = np.argmax(probsNew)

            ellAreaNew = np.pi * np.sqrt(np.linalg.det(covNew[kNew,0:2,0:2]))
            ellAreaOld = np.pi * np.sqrt(np.linalg.det(covOld[kOld,0:2,0:2]))
            if self.snapshotOnly:
                outProbs[kOld] = scoreOld * probsOld[kOld] * ellAreaNew / \
                    (scoreNew * probsNew[kNew] * ellAreaOld + scoreOld * probsOld[kOld] * ellAreaNew)                
            else:
                outProbs[kOld] = probsOld[kOld] * ellAreaNew / (probsNew[kNew] * ellAreaOld + probsOld[kOld] * ellAreaNew)
            outProbs[kNew] = 1.0 - outProbs[kOld]
            posNew[kOld,:] = posNew[kNew,:]
            covNew[kOld,:,:] = covNew[kNew,:,:]
            posOld[kNew,:] = posOld[kOld,:]
            covOld[kNew,:,:] = covOld[kOld,:,:]
            return outProbs

        newIsConsistent = self.posesAreConsistent(posNew, covNew, probsNew)
        oldIsConsistent = self.posesAreConsistent(posOld, covOld, probsOld)

        if newIsConsistent and oldIsConsistent:
            if not self.snapshotOnly and np.sum(probsCurrent) < 0.01:
                # first observation for this timestamp... pose change possible so err on the side of latest observation
                return probsNew
            else:
                outProbs = probsCurrent * probsNew
                outProbs /= np.sum(outProbs)
                return outProbs

        typeToUse = -2 * np.ones((self.numPoseTypes,2),np.int)

        typeToUse[commonTypes,:] = np.expand_dims(commonTypes, axis=1)
        if not self.snapshotOnly and oldIsConsistent:
            newTypesOnly = np.where(np.logical_and(probsNew > 0.01, probsOld < 0.01))[0]
            typeToUse[newTypesOnly,0] = newTypesOnly
            typeToUse[newTypesOnly,1] = -1 # fall back to the consolidated version

        if 0 and self.snapshotOnly and newIsConsistent:
            # need to be symmetric in terms of order since it is the same time stamp
            oldTypesOnly = np.where(np.logical_and(probsNew < 0.01, probsOld > 0.01))[0]
            typeToUse[oldTypesOnly,1] = oldTypesOnly
            typeToUse[oldTypesOnly,0] = -1 # fall back to the consolidated version

        zVals = 1000.0 * np.ones(Types.size)
        for k2, k in enumerate(Types):
            if np.all(typeToUse[k,:] >= -1):
                Diff = np.array([posNew[typeToUse[k,0],:]-posOld[typeToUse[k,1],:]])
                combCov = covNew[typeToUse[k,0],:,:]+covOld[typeToUse[k,1],:,:]
                zVals[k2] = np.matmul(Diff,np.matmul(np.linalg.inv(combCov),Diff.transpose()))
        Probs = 1.0 - self.myChiDistGuy.cdf(zVals, 3 * np.ones(Types.size,np.int))
        outProbs[Types] = Probs * (Probs > 0.35 * np.max(Probs)).astype(np.float)
        outProbs[Types] *= probsNew[Types] #+ probsOld[Types]
        #outProbs[Types] *= probsNew[Types] + probsOld[Types]

        outProbs /= np.sum(outProbs)
        return outProbs

    def update_instance_kalman(self, j, timeStamp, posAndHeight, posHeightCov, poseTypeProbs, colorFeats, colorCov, hasColor, channelNum = None, obsInChannel=None, bodyOrientation=[], Score = []):
        # first position
        if j == 4:
            d = 1

        if channelNum is None:
             # receiving data from all channels (could be a single one still if there only is one)
            self.numObs[j, :] += obsInChannel
            # blend new color stats with old
            for k in range(self.numChannels):
                if hasColor[k]:

                    if self.numColorObs[j,k] < self.nmFrmsHrznColor:
                        numOldSamples = self.numColorObs[j,k]
                        self.numColorObs[j,k] += 1
                    else:
                        numOldSamples = self.nmFrmsHrznColor - 1
                    #numOldSamples = min(self.nmFrmsHrznColor,self.numColorObs[j,k])
                    #self.numColorObs[j,k] += 1
                    mixOfNew = 1.0 / (1.0 + numOldSamples)
                    mixOfOld = 1.0 - mixOfNew
                    oldVals  = self.colorFeats[j,k,:].copy()
                    self.colorFeats[j,k,:] *= mixOfOld
                    self.colorFeats[j,k,:] += mixOfNew * colorFeats[k,:]
                    self.colorCov[j,k,:,:] *= mixOfOld
                    self.colorCov[j,k,:,:] += mixOfNew * colorCov[k,:,:]
                    #self.colorCov[j,k,:,:] += mixOfNew * np.multiply(colorFeats[k,:]-self.colorFeats[j,k,:] ,np.expand_dims(colorFeats[k,:]-self.colorFeats[j,k,:] ,axis=1)) + mixOfOld * np.multiply(self.colorFeats[j,k,:] -oldVals,np.expand_dims(self.colorFeats[j,k,:] -oldVals,axis=1)) 
                else:
                    self.numColorObs[j,k] = max(0,self.numColorObs[j,k]-1)
        else:
            if 1:
                self.numObs[j, channelNum] += 1
                # blend new color stats with old
                if hasColor:

                    if self.numColorObs[j,channelNum] < self.nmFrmsHrznColor:
                        numOldSamples = self.numColorObs[j,channelNum]
                        self.numColorObs[j,channelNum] += 1
                    else:
                        numOldSamples = self.nmFrmsHrznColor - 1

                    mixOfNew = 1.0 / (1.0 + numOldSamples)
                    mixOfOld = 1.0 - mixOfNew
                    oldVals  = self.colorFeats[j,channelNum,:].copy()
                    self.colorFeats[j,channelNum,:] *= mixOfOld
                    self.colorFeats[j,channelNum,:] += mixOfNew * colorFeats
                    self.colorCov[j,channelNum,:,:] *= mixOfOld
                    self.colorCov[j,channelNum,:,:] += mixOfNew * colorCov
                else:
                    self.numColorObs[j,channelNum] = max(0,self.numColorObs[j,channelNum]-1)

            else:
                # receiving data from single channel
                self.numObs[j, channelNum] += 1  
                # since color is by channel, we don't expect ever to have to update, just add
                if hasColor:
                    self.numColorObs[j,channelNum] = 1
                    self.colorFeats[j,channelNum,:] = colorFeats
                    self.colorCov[j,channelNum,:,:] = colorCov       

        if bodyOrientation != []:
            self.bodyOrientation[j,:] = self.bodyOrientation[j,:] * bodyOrientation
            self.bodyOrientation[j,:] = self.bodyOrientation[j,:] / np.sum(self.bodyOrientation[j,:])
                    
        # now to the positions. First, do an update of the pose probabilities, in case there are more non-zero ones
        oldProbs = self.poseTypeProb[j,:].copy()
            
        if self.snapshotOnly:            
            self.poseTypeProb[j,:] = self.reconcileProbabilities(posAndHeight, posHeightCov, poseTypeProbs, self.posAndHeight[j,:,:],self.posCov[j,:,:,:],self.poseTypeProb[j,:],self.poseTypeProb[j,:],Score,self.Score[j])
            self.Score[j] = 1.0 - (1.0 - Score) * (1.0 - self.Score[j])
        else:
            self.poseTypeProb[j,:] = self.reconcileProbabilities(posAndHeight, posHeightCov, poseTypeProbs, self.posAndHeightTemp[j,:,:],self.posCovTemp[j,:,:,:],self.poseTypeProb[j,:],self.poseTypeProbCurrent[j,:])
            self.lastTimeStampPerPerson[j] = timeStamp
        for k in np.where(self.poseTypeProb[j,:] > 0.01)[0]:
            # We will assume that the probabilities have already been updated in terms of eliminating ones that are 
            # mutually exclusive 
            # so, either they are both active, none of them active, or old one zero and new one non-zero
            if oldProbs[k] < 0.01:
                # it's a new thing that this pose gets any probability
                self.KalmanState[j,k,:] = 0.0 
                self.KalmanState[j,k,0:3] = posAndHeight[k,:] 
                self.KalmanCov[j,k,:,:] = 0.0
                self.KalmanCov[j,k,0:3,0:3] = posHeightCov[k,:,:]
                if self.postureMobile[k]:
                    self.KalmanCov[j,k,3,3] = self.veloVar
                    self.KalmanCov[j,k,4,4] = self.veloVar

            elif self.poseTypeProb[j,k] > 0.01:
                if self.snapshotOnly:
                    invCov1 = np.linalg.inv(self.posCov[j,k,:,:])
                    invCov2 = np.linalg.inv(posHeightCov[k,:,:])

                    Cov3 = np.linalg.inv(invCov1 + invCov2)
                    self.posAndHeight[j,k,:] = np.squeeze(np.matmul(Cov3,np.matmul(invCov1,np.expand_dims(self.posAndHeight[j,k,:],axis=1))) + 
                                        np.matmul(Cov3,np.matmul(invCov2,np.expand_dims(posAndHeight[k,:],axis=1))))
                    self.posCov[j,k,:,:] = Cov3    
                else:
                    # Finally: The Kalman update!
                    y = np.expand_dims(posAndHeight[k,:] - self.KalmanStateUpdate[j,k,0:3],axis=1)
                    Sk = np.matmul(self.H,np.matmul(self.KalmanCovUpdate[j,k,:,:],self.H.transpose())) + posHeightCov[k,:,:]
                    Kk = np.matmul(np.matmul(self.KalmanCovUpdate[j,k,:,:],self.H.transpose()), np.linalg.inv(Sk))
                    self.KalmanState[j,k,:] = self.KalmanStateUpdate[j,k,:] + np.matmul(Kk,y).squeeze()
                    self.KalmanCov[j,k,:,:] = np.matmul(np.eye(5)-np.matmul(Kk,self.H), self.KalmanCovUpdate[j,k,:,:])                    
            
            # since we may get another observation for the same time stamp, need to update the predict step also
            self.KalmanStateUpdate[j,k,:] = self.KalmanState[j,k,:]
            self.KalmanCovUpdate[j,k,:,:] = self.KalmanCov[j,k,:,:]

            # Update legacy data (for now)
            self.posAndHeightTemp[j,k,:] = self.KalmanStateUpdate[j,k,0:3]
            self.posCovTemp[j,k,:,:] = self.KalmanCovUpdate[j,k,0:3,0:3]

            self.posAndHeight[j,k,:] = self.posAndHeightTemp[j,k,:]
            self.posCov[j,k,:,:] = self.posCovTemp[j,k,:,:] 

        self.computeAggregatePosInternal(np.array([j]))
        if not self.snapshotOnly:
            self.poseTypeProbCurrent[j,:] = self.poseTypeProb[j,:]
            if channelNum != None:
                self.nConseqMisses[j,channelNum] = 0
                self.nConseqHits[j,channelNum] += 1   
            else:
                self.nConseqMisses[j,obsInChannel>0] = 0
                self.nConseqMisses[j,obsInChannel==0] += 1
                self.nConseqHits[j,obsInChannel>0] += 1   
                self.nConseqHits[j,obsInChannel==0] = 0 
            self.Track[j].addObservation(posAndHeight[-1,0:2], timeStamp)

    def report_evidence_of_distinctness(self, Inds):
        # accumulate evidence that a set of persons are distinct, i.e. not duplicates. This typically occurs when a set of characters are seen 
        # in the frame and camera view
        for i2, i in enumerate(Inds):
            for j2, j in enumerate(Inds[0:i2]):
                self.probDuplicate[i,j] = min(-1, self.probDuplicate[i,j] - 1)  # Letting a observation of distinctness trump evidence of duplicateness
                self.probDuplicate[j,i] = self.probDuplicate[i,j]

    def report_evidence_of_duplicates(self, Inds):
        # accumulate evidence that two (or more) persons are duplicates of one another 
        for i2, i in enumerate(Inds):
            for j2, j in enumerate(Inds[0:i2]):
                self.probDuplicate[i,j] += 1
                self.probDuplicate[j,i] = self.probDuplicate[i,j]

    def detect_and_remove_duplicates(self):
        # accumulate evidence that two (or more) persons are duplicates of one another 
         # look for positive values -> likely the same point...
        IndsB = self.Inds
        candInds = np.argwhere(np.triu(self.probDuplicate[np.ix_(IndsB,IndsB)],1) >= 1)

        wasRemoved = np.zeros(0,np.int)
        replacement = np.zeros(0,np.int)
        for k in range(candInds.shape[0]):   
            if not np.any(np.isin(candInds[k,:],wasRemoved)):
                winner = self.mergePeople(IndsB[candInds[k,0]], IndsB[candInds[k,1]])
                wasRemoved = np.append(wasRemoved,IndsB[candInds[k,np.where(np.isin(IndsB[candInds[k,:]],winner, invert=True))[0]]])
                replacement = np.append(replacement,winner)
        return wasRemoved, replacement       

    def add_instance(self, posAndHeight, posHeightCov, poseTypeProbs, prob_human, timeStamp, colorFeats, colorCov, hasColor, channelNum =None, obsInChannel=None,bodyOrientation=[],Score=[]):
        numHumans = self.Inds.shape[0]
        if (numHumans >= self.numAlloc):
             # allocate more humans...
             self.reallocHumans(self.numAlloc + self.allocBlockSize)
        # now find which slot/index to use
        i = 0
        while self.slotUsed[i]:
            i += 1
        self.Inds = np.sort(np.append(self.Inds, i))    
        self.slotUsed[i] = True
        if channelNum is None:
            # receiving data from all channels (could be a single one still if there only is one)
            self.numObs[i,:] = obsInChannel
            self.numColorObs[i,:] = hasColor.astype(np.int)
            self.colorFeats[i,hasColor,:] = colorFeats[hasColor,:]
            self.colorCov[i,hasColor,:,:] = colorCov[hasColor,:,:]
        else:
            # receiving data from single channel
            self.numObs[i,:] = 0
            self.numObs[i,channelNum] = 1
            self.numColorObs[i,:] = 0
            if hasColor:
                self.numColorObs[i,channelNum] = 1
                self.colorFeats[i,channelNum,:] = colorFeats
                self.colorCov[i,channelNum,:,:] = colorCov
                  
        if bodyOrientation != []:
            self.bodyOrientation[i,:] = bodyOrientation
        else:
            self.bodyOrientation[i,:] = 1.0 / self.numAngleBins

        self.poseTypeProb[i,:] = poseTypeProbs
        self.temp_ids[i] = self.uniqishStringID(i)

        if self.snapshotOnly:
            self.Score[i] = Score
        else:
            self.nConseqMisses[i,:] = 0
            self.nConseqHits[i,:] = 0
            self.nConseqHits[i,channelNum] = 1
            self.lastTimeStampPerPerson[i] = timeStamp
            self.poseTypeProbCurrent[i,:] = poseTypeProbs
            self.poseTypeProbPrev[i,:] = 0.0            

        if not self.snapshotOnly:
            self.KalmanState[i,:,:] = 0.0
            self.KalmanCov[i,:,:,:] = 0.0

        for kk in np.where(poseTypeProbs > 0.01)[0]:
            self.posAndHeight[i,kk,:] = posAndHeight[kk,:]
            self.posCov[i,kk,:,:] = posHeightCov[kk,:,:]
            if not self.snapshotOnly:
                self.posAndHeightTemp[i,kk,:] = posAndHeight[kk,:]            
                self.posCovTemp[i,kk,:,:] = posHeightCov[kk,:,:]
                self.KalmanState[i,kk,0:3] = posAndHeight[kk,:] 
                self.KalmanCov[i,kk,0:3,0:3] = posHeightCov[kk,:,:]
                if self.postureMobile[kk]:
                    self.KalmanCov[i,kk,3,3] = self.veloVar
                    self.KalmanCov[i,kk,4,4] = self.veloVar
                # since we may get another observation for the same time stamp, need to update the predict step also
                self.KalmanStateUpdate[i,kk,:] = self.KalmanState[i,kk,:]
                self.KalmanCovUpdate[i,kk,:,:] = self.KalmanCov[i,kk,:,:]
        
        self.computeAggregatePosInternal(np.array([i]))
        if not self.snapshotOnly:
            if self.doFaceID:
                self.db_ids[i] = None            
                self.names[i] = None
                self.face_images[i] = None
                self.face_detected[i] = False
            self.Track[i].newTrack(i, self.numReincarnations[i], posAndHeight[-1,0:2], timeStamp)


            



        return i

    def reallocHumans(self, nToAlloc):
        N = nToAlloc - self.numAlloc
        self.numAlloc = nToAlloc
        self.posAndHeight.resize((nToAlloc,self.numPoseTypes,self.numPosDims))
        self.poseTypeProb.resize((nToAlloc,self.numPoseTypes))
        self.posCov.resize((nToAlloc,self.numPoseTypes,self.numPosDims,self.numPosDims))
        self.bodyOrientation.resize((nToAlloc,self.numAngleBins)) 
        self.colorFeats.resize((nToAlloc,self.numChannels,self.numColorFeats), refcheck=False)
        self.colorCov.resize((nToAlloc,self.numChannels,self.numColorFeats,self.numColorFeats), refcheck=False)
        self.numColorObs.resize((nToAlloc,self.numChannels), refcheck=False) 
        self.slotUsed.resize(nToAlloc)
        self.numReincarnations.resize(nToAlloc)
        self.numObs.resize((nToAlloc,self.numChannels))
        self.temp_ids += [None] * N
        if self.snapshotOnly:
           self.Score.resize(nToAlloc)
        else:
            self.posAndHeightTemp.resize((nToAlloc,self.numPoseTypes+1,self.numPosDims))
            self.poseTypeProbPrev.resize((nToAlloc,self.numPoseTypes))
            self.poseTypeProbCurrent.resize((nToAlloc,self.numPoseTypes)) 
            self.poseTypeProbVelo.resize((nToAlloc,self.numPoseTypes))
            self.posCovTemp.resize((nToAlloc,self.numPoseTypes+1,self.numPosDims,self.numPosDims))            
            self.lastTimeStampPerPerson.resize(nToAlloc)
            self.probDuplicate.resize((nToAlloc,nToAlloc))
            self.nConseqMisses.resize((nToAlloc,self.numChannels))
            self.nConseqHits.resize((nToAlloc,self.numChannels))    
            
            self.KalmanState.resize((nToAlloc,self.numPoseTypes,5))
            self.KalmanCov.resize((nToAlloc,self.numPoseTypes,5,5))
            self.KalmanStateUpdate.resize((nToAlloc,self.numPoseTypes,5))
            self.KalmanCovUpdate.resize((nToAlloc,self.numPoseTypes,5,5))

            for ii in range(N):
                self.Track.append(track())
            
            if self.doFaceID:
                self.db_ids += [None] * N      
                self.names += [None] * N
                self.face_images += [None] * N
                self.face_detected.resize(nToAlloc)

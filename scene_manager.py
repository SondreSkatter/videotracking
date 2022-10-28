import numpy as np
#from scipy import ndimage
import chidistdf
import cv2
import multiprocessing
from custom_queue import Queue
import time
from scipy.optimize import linear_sum_assignment
import tracks, camera
import bounding_box_stuff
import posenet_helper, graph_helper
import static_objects_pose_est
from face_api_client import FaceAPI
from face_request_worker import RequestMessage
# this guys only deals with 2D coordinates in meters (and 3D if you count a person's height)

numAngleBins = 20  # even number please
doOrientation = False
faceIDinterval = 1
numColorFeats = 2

class scene_manager:
    def __init__(self, channels, trckCfg, face_input_q=None, face_output_q=None, unknownHeight=False):
        # CONFIG params
        self.track_in_q = [] # just dummy 
        self.det_threshold_add_new = 0.4
        self.channels = channels
        self.includePosInfo = trckCfg["globalGT"] != []
        self.minNumSightings = trckCfg["minNumSightings"]
        self.numChannels = len(channels)
        self.usePersistedStaticObjects = trckCfg["usePersistedStaticObjects"]
        self.detectStaticObjects = trckCfg["detectStaticObjects"]
        self.exportStaticObjects = (self.detectStaticObjects and trckCfg["showStaticObjects"])
        self.exportVisitorMap = (self.detectStaticObjects and trckCfg["showVisitorMap"])
        self.doFaceID = trckCfg["doFaceID"]       

        numAlloc = 150
        self.numAngleBins = numAngleBins
        self.faceIDinterval = faceIDinterval
        self.doOrientation = doOrientation
        self.numColorFeats = numColorFeats
        self.poseHelper = posenet_helper.posenet_helper()

        self.myCamMappers = []

        for i, c in enumerate(channels):
            self.myCamMappers.append(camera.camera(c.imSize, c.calScale, c.calib_path, c.useGeometricCenterAsRotAxis))
        if self.detectStaticObjects:
            self.static_objects = []
            for i, c in enumerate(channels):
                self.static_objects.append(static_objects_pose_est.static_objects_pose_est(c.imSize, c.calib_path, c.camName, trckCfg["usePersistedStaticObjects"], self.poseHelper))

        self.colPrbFn = self.poseHelper.KLtoP1eqs
        self.postureMobile = self.poseHelper.postureMobile

        self.numPoseTypes = self.postureMobile.shape[0]
        self.myChiDistGuy = chidistdf.chidist()
        self.myTracks = tracks.tracks(self.numChannels,self.myChiDistGuy, self.numColorFeats, self.postureMobile, numAlloc, self.numAngleBins, False, self.doFaceID, unknownHeight)            
        # also add another such container object for representing snapshot, i.e. consolidating multiple observations (cameras) at the same time point
        self.myTempTracks =  tracks.tracks(self.numChannels,self.myChiDistGuy, self.numColorFeats, self.postureMobile, numAlloc, self.numAngleBins, True, False, unknownHeight) 

        self.FA0 = 0.0015 # minimum Pfa rate (think mannequin)
        self.maxConseqMisses = 10 
        self.minProb = 0.005
        self.minLogRatio = np.log(1.3)

        self.oldAlgo = True
        if self.oldAlgo:
            self.enforceConsistency = True
            self.minProb = 0.5 * np.array((0.01,0.01,0.01))
            self.minLogRatios = np.log(np.array((1.4,1.3,1.5)))
            testVals = np.arange(2.0,80.0)
            testProbs = 1.0 - self.myChiDistGuy.cdf(testVals, 3*np.ones(testVals.shape,np.int))
            self.z_consist_threshold = testVals[np.max(np.where(testProbs > 0.05)[0])]

        if 0:
            testVals = np.arange(2.0,80.0)
            testVals = np.linspace(1.0,30.0,500)
            testProbs = 1.0 - self.myChiDistGuy.cdf(testVals, 3*np.ones(testVals.shape,np.int))
            self.z_threshold = testVals[np.max(np.where(testProbs > self.minProb)[0])]
            self.distinct_z_threshold = testVals[np.max(np.where(testProbs > 0.75)[0])]
        self.z_threshold = 12.0
        self.distinct_z_threshold_intra_frame = 2.0# 1.3 #2.0
        self.distinct_z_threshold_tracking = 2.0


        self.instrumentFNstats = False
        if self.doFaceID:
            self.total_face_jobs = 0
            self.successful_face_reqs = 0
            self.unsuccessful_face_reqs = 0
            self.face_output_queue = face_output_q
            self.face_input_queue = face_input_q

            self.total_response_time = 0
            self.log_path = './api/stats/client_logs/response.log'
            self.wait_for_api = 1 #wait for api requests for total network latency benchmark

    def shutdown(self):
        if self.detectStaticObjects:
            for i, c in enumerate(self.static_objects):
                c.save_static_objects()
        if self.doFaceID:
            print('# face jobs issued: ', self.total_face_jobs)
            print('# of successful identification requests: ', self.successful_face_reqs)
            print('# of requests with unidentifiable faces: ', self.unsuccessful_face_reqs)
            print('average response time : ', self.total_response_time/self.total_face_jobs)

    def identifyLikelyDuplicates(self, grphEdges, posCovA):
        # analyze the second dimension to see if there is a n-1 relationship to the first dimension
        subsets = []

        rowsums = np.sum(grphEdges.astype(np.int), axis=1)
        singletonsCols = np.sum(grphEdges.astype(np.int), axis=0) == 1
        rowsumsSingletons = np.sum(grphEdges[:,singletonsCols].astype(np.int), axis=1)
        rowsOfInterest = np.where(np.logical_and(rowsums == rowsumsSingletons, rowsums > 1))[0]

        for i in rowsOfInterest:
            # also make sure this is not a point with large covariance, connecting two disjoint B points
            if np.trace(posCovA[i,-1,:,:]) < np.square(1.5):
                subsets.append(np.where(grphEdges[i,:])[0])
        return subsets

    def compareOrientions(self, orA, orB):
        if 0:
            maxTurnDegrees = 60.0
            maxTurnBins = int(np.ceil(maxTurnDegrees / 360.0 * self.numAngleBins))
            Conv = ndimage.convolve(orA,orB[::-1],mode='wrap')
            if self.instrumentFNstats:
                return Conv
            maxHome = np.max(Conv[(self.numAngleBins//2-maxTurnBins-1):(self.numAngleBins//2+maxTurnBins)])
            globMax = np.max(Conv)
            if globMax > maxHome:
                # looks like it might be different
                return np.square(globMax / maxHome)
            else:
                return 1.0
        return 0.5

    def computeLikelihoods(self,zThres,IndsA, IndsB, revLookUpA, revLookUpB, posAndHghtA,posCovA, poseProbA, posAndHghtB,posCovB, poseProbB,
        hasColorA=None, colorFeatsA=None, colorCovA=None, hasColorB=None, colorFeatsB=None, colorCovB=None, OrientA=None, OrientB=None,
        AandBisSame=False):         
        goodIndsA = np.argwhere(poseProbA[IndsA,:] > 0.01)
        goodIndsB = np.argwhere(poseProbB[IndsB,:] > 0.01)
        goodIndsA[:,0] = IndsA[goodIndsA[:,0]]
        goodIndsB[:,0] = IndsB[goodIndsB[:,0]]

        muXdiff = posAndHghtB[goodIndsB[:,0],goodIndsB[:,1],0] - np.expand_dims(posAndHghtA[goodIndsA[:,0],goodIndsA[:,1],0],axis=1)
        muYdiff = posAndHghtB[goodIndsB[:,0],goodIndsB[:,1],1] - np.expand_dims(posAndHghtA[goodIndsA[:,0],goodIndsA[:,1],1],axis=1)
        
        totDistSq = np.square(muXdiff) + np.square(muYdiff)

        posVarA = np.expand_dims(posCovA[goodIndsA[:,0],goodIndsA[:,1],0,0] + posCovA[goodIndsA[:,0],goodIndsA[:,1],1,1], axis=1) 

        candInds2 = np.argwhere(np.logical_or(totDistSq < np.square(4.0), totDistSq - 4 * posVarA < 0.0))
        candInds = np.zeros(candInds2.shape, np.int)      
  
        if candInds.shape[0] > 0:
            candInds[:,0] = goodIndsA[:,0][candInds2[:,0]]
            candInds[:,1] = goodIndsB[:,0][candInds2[:,1]]
            _, ic = np.unique(candInds,axis=0,return_index=True)
            candInds = candInds[ic,:]
            if AandBisSame:
                # this means we're comparing a set to itself. So we can skip the upper triangle of the matrix
                # we have to be careful though since the quick metric used above to filter is not symmetric in A & B
                isDistinct = np.ones((IndsA.size,IndsA.size),np.bool)
                isDistinct[revLookUpA[candInds[:,0]],revLookUpA[candInds[:,1]]] = False
                isDistinct = np.logical_and(isDistinct,isDistinct.transpose())
                isDistinct[np.arange(IndsA.size),np.arange(IndsA.size)]= True
                candInds = IndsA[np.argwhere(isDistinct==False)]
                lowTriInds = np.where(candInds[:,0] < candInds[:,1])[0]
                candInds = candInds[lowTriInds,:]

        NCands = candInds.shape[0]
        zValue = np.zeros(NCands, np.float)
        DoF = 3 * np.ones(NCands, np.int)
        areaFactors = np.ones(NCands, np.float)

        for kk in range(NCands):
            i = candInds[kk,0]
            j = candInds[kk,1]

            probProd = poseProbA[i,:] * poseProbB[j,:]
            commonInds = np.where(probProd > 0.0001)[0]

            if commonInds.size == 0:
                indPair = np.array([[np.argmax(poseProbA[i,:])],[np.argmax(poseProbB[j,:])]])
            else:
                indPair = np.vstack((commonInds,commonInds))
            bestL = 0.0
            zValue[kk] = 1E16
            bestZratio = 1E30
            for qq in range(indPair.shape[1]):                 
                Diff = np.array([posAndHghtB[j,indPair[1,qq],:]-posAndHghtA[i,indPair[0,qq],:]])
                combCov = posCovB[j,indPair[1,qq],:,:]+posCovA[i,indPair[0,qq],:,:]
                zVal = np.matmul(Diff,np.matmul(np.linalg.inv(combCov),Diff.transpose()))
                zVal2D = np.matmul(Diff[:,0:2],np.matmul(np.linalg.inv(combCov[0:2,0:2]),Diff[:,0:2].transpose()))
                if zVal2D < 0.6 * zVal:
                    zVal = zVal2D
                    DoF[kk] = 2

                if AandBisSame:
                    distinct = zVal > zThres
                    isDistinct[revLookUpA[i],revLookUpB[j]] = distinct
                    isDistinct[revLookUpB[j],revLookUpA[i]] = distinct                        
                else:
                    if zVal < zThres:
                        personalSpaceOfManSqMeters = 1.75 
                        # compute the area of the error ellipse (basically, if computing the eigenvalues and the area of 
                        # the ellipse defined by it, we get the answer below
                        areaOfErrEllipse = np.pi * np.sqrt(combCov[0,0] * combCov[1,1] - combCov[1,0] * combCov[0,1])
                        areaFactor = personalSpaceOfManSqMeters / max(personalSpaceOfManSqMeters,areaOfErrEllipse)

                        zRatio = zVal / areaFactor
                        if zRatio < bestZratio:
                            bestZratio = zRatio
                            zValue[kk] = zVal
                            areaFactors[kk] = areaFactor
    
        newCandInds = np.where(zValue < zThres)[0]
        NCands = newCandInds.size
        candInds = candInds[newCandInds,:]
        if 1:
             l1 = areaFactors[newCandInds] * (1.0 - self.myChiDistGuy.cdf(zValue[newCandInds], DoF[newCandInds]))
        else:
            l1 = areaFactors[newCandInds] * (1.0 - self.myChiDistGuy.cdf(zValue[newCandInds], 3*np.ones(NCands,np.int)))
        l1Spatial = l1.copy()
        if 1 and not hasColorA is None:
            KL = np.zeros((NCands,self.numChannels), np.float)
            P1ToP0col = np.ones((NCands,self.numChannels), np.float) # likelihood ratio of beign different vs same person

            # http://math.bme.hu/~marib/tobbvalt/tv5.pdf
            # now look for cases where is a need for color info in order to decide...
            for kk in range(NCands):
                i = candInds[kk,0]
                j = candInds[kk,1]
                for c in np.where(hasColorA[revLookUpA[i],:] * hasColorB[revLookUpB[j],:])[0]:
                    Cov1 = colorCovB[j,c,:,:]
                    Cov0 = colorCovA[i,c,:,:]
                    det1 = np.linalg.det(Cov1)
                    det0 = np.linalg.det(Cov0)
                    if det1 > 1E-16 and det0 > 1E-16: 
                        Diff = np.array([colorFeatsA[i,c,:] - colorFeatsB[j,c,:]])
                        invCov1 = np.linalg.inv(Cov1)
                        # compute KL divergence
                        KL[kk,c] = 0.5 * (np.trace(np.matmul(invCov1,Cov0)) + np.matmul(Diff, np.matmul(invCov1, Diff.transpose())) - self.numColorFeats + np.log(det1/det0))   
                        #P1ToP0col[kk,c] = np.interp(KL[kk,c],self.colPrbFn[c,:,0],self.colPrbFn[c,:,1],right=self.colPrbFn[c,-1,1])
                        P1ToP0col[kk,c] = np.interp(KL[kk,c],self.colPrbFn[:,0],self.colPrbFn[:,1])
                        l1[kk] /= l1[kk]  + (1 - l1[kk]) * P1ToP0col[kk,c] # a Bayesian update for the probability of matching...
                if self.doOrientation:
                    # let's do orientation too!
                    lkRatioIsDifferent = self.compareOrientions(OrientA[i,:], OrientB[j,:])
                    l1[kk] /= l1[kk]  + (1 - l1[kk]) * lkRatioIsDifferent
            d = 1
        if AandBisSame:
            return isDistinct
        return l1, candInds

    def getCamPose(self, channelNum):
        return self.myCamMappers[channelNum].trans, self.myCamMappers[channelNum].horRotMat, self.myCamMappers[channelNum].leftTan, self.myCamMappers[channelNum].rightTan, self.myCamMappers[channelNum].btmTan, self.myCamMappers[channelNum].topTan

    def getFNprobs(self, channelNum, IndsB):
        
        camPos, rotMat, leftTan, rightTan, btmTan, topTan = self.getCamPose(channelNum)

        if self.oldAlgo:
            myTracks = self.myTracks  
            posAndHeight = myTracks.posAndHeightTemp
            posCov = myTracks.posCovTemp
        else:
            myTracks = self.myTempTracks  
            posAndHeight = myTracks.posAndHeight
            posCov = myTracks.posCov

        Inds = myTracks.Inds
        relPos = posAndHeight[Inds,-1,0:2] - np.expand_dims(camPos[0:2],axis=0)

        distFromCamera = np.linalg.norm(relPos,axis=1)

        # now rotate the positions as well
        # make it so x = 0 is the central direction
        relPos = np.matmul(rotMat,relPos.transpose())

        # also estimate the angular "spread" of the person...
        personHalfWidth = 0.2

        varX = np.zeros(Inds.shape[0])
        for j2, j in enumerate(Inds):
            kInd = np.argmax(myTracks.poseTypeProb[j,:])
            varX[j2] = posCov[j,kInd,0,0] * np.square(rotMat[0,0]) + \
                posCov[j,kInd,1,1] * np.square(rotMat[0,1]) + 2.0 * posCov[j,kInd,0,1] * rotMat[0,0] * rotMat[0,1] +np.square(personHalfWidth)
        varTan = varX / np.square(distFromCamera)

        Tan = relPos[0,:] / relPos[1,:]

        heightTanTop = (posAndHeight[Inds,-1,2] - camPos[2]) / distFromCamera

        heightTanBtm = (0 - camPos[2]) / distFromCamera
        numVertSectors = 36
        numHorSectors = 96
        horSectorBinSize = (rightTan - leftTan) / (numHorSectors - 1)
        horSectorBins = np.linspace(leftTan, rightTan, numHorSectors) + horSectorBinSize / 2        

        topInd = (np.floor(numVertSectors * (heightTanTop - btmTan) / (topTan - btmTan))).astype(np.int) 
        topIndCropped = topInd.copy()
        topIndCropped[topIndCropped >= numVertSectors-1] = numVertSectors-1
        btmInd = (np.floor(numVertSectors * (heightTanBtm - btmTan) / (topTan - btmTan))).astype(np.int) 
        btmIndCropped = btmInd.copy()
        btmIndCropped[btmIndCropped < 0] = 0
        vertSizeFraction = (heightTanTop - heightTanBtm) / (topTan - btmTan)

        inScope = np.logical_and(np.logical_and(relPos[1,:] > 0.0,topIndCropped>btmIndCropped), np.logical_and(btmInd < numVertSectors, topInd >= 0))  
        goodInds = np.where(inScope)[0]
        
        percInFrame = np.zeros(IndsB.shape[0],np.float)
        vertSizeFractions = np.zeros(IndsB.shape[0],np.float) 
        visibilities = np.zeros(IndsB.shape[0],np.float) 
        bgCoverage = np.zeros((numVertSectors,numHorSectors), np.float)
        Constant = horSectorBinSize / np.sqrt(2 * np.pi)
        # Now, we will accumulate people-in-front-of for each Sector
        for i in goodInds[np.argsort(distFromCamera[goodInds])]:            
            # first compute the forground coverage
            fgCoverage = np.zeros((numVertSectors,numHorSectors), np.float)
            ProbProfile = np.exp(-0.5 * np.square(horSectorBins - Tan[i]) / varTan[i]) * Constant / np.sqrt(varTan[i]) # Gaussian pdf, with bin width multiplied in
            totProb = np.sum(ProbProfile)
            
            if totProb > 1.0:
                ProbProfile /= totProb
                totProb = np.sum(ProbProfile)

            fgCoverage[btmIndCropped[i]:topIndCropped[i],:] = ProbProfile  #* np.sqrt(varX[i]) / personHalfWidth  # normalizing so that it rather represents probability of visibility for each cell
            k = np.where(IndsB == Inds[i])[0]
            if k.size == 1:
                percInFrame[k] = totProb * (topIndCropped[i] - btmIndCropped[i]) / (topInd[i] - btmInd[i])
                vertSizeFractions[k] = vertSizeFraction[i]
                if np.sum(fgCoverage) > 0.001:
                    visibility = np.sum(fgCoverage * (1 - bgCoverage)) / np.sum(fgCoverage)
                    visibilities[k] = visibility

            bgCoverage = 1.0 - (1 - fgCoverage) * (1 - bgCoverage)

        if self.instrumentFNstats:
            return visibilities, percInFrame, vertSizeFractions

        # Model developed based on stats collected from wildtrtack data
        B0 = 19.6976
        B = np.array([[-7.9696, -13.2605, -4.03416, 0.7697]])

        pFNB = np.zeros(IndsB.size)
        pFNB[:] = 1.0 / (1.0 + np.exp(-B0 - np.squeeze(np.matmul(B,np.vstack((percInFrame, visibilities, vertSizeFractions, np.square(vertSizeFractions)))))))

        return pFNB
               
    def match_humans_in_frame_only(self, channels, goodInds, timeStamps, posAndHght, posHeightCov, poseTypeProb, Score, colorFeats,
        colorCov, hasColorA, Orientations, unitVectorsLoc, keyPtsScrLoc):
        
        self.myTempTracks.clear() # fresh start
        N = channels.shape[0]
        
        nA = np.zeros(N,np.int)
        goodMatchesA = [None] * N
        goodMatchesB = [None] * N
        for c in range(N):
            nA[c] = Score[c].shape[0]

        revLookUpA = np.zeros(np.max(nA)+1,np.int)
        revLookUpB = np.zeros(self.myTracks.numAlloc, np.int)

        # in these loops we can match to gallery, and do an update inline, and we can also add to gallery.
        # we can't delete or merge though. that needs to happen afterwards            
        for c in range(N):                  
            IndsB = self.myTempTracks.Inds 
            if self.myTempTracks.numAlloc > revLookUpB.size:
                revLookUpB.resize(self.myTracks.numAlloc)
            
            IndsA = goodInds[c] 
            na = IndsA.size
            if na == 0:
                continue
            revLookUpA[IndsA] = np.arange(na) 

            # First: establish distinctness of the new data, i.e. are pairs of people mutually spatially resolved or not
            isDistinct = self.computeLikelihoods(self.distinct_z_threshold_intra_frame, IndsA, IndsA, revLookUpA, revLookUpA, posAndHght[c],posHeightCov[c],
                poseTypeProb[c], posAndHght[c],posHeightCov[c],poseTypeProb[c],AandBisSame=True)
                               
            revLookUpB[IndsB] = np.arange(IndsB.size)
            toAdd = np.zeros(0, np.bool)
            pPost = np.zeros((IndsA.size,IndsB.size), np.float)    
            if IndsB.size > 0:    
                l1, candInds = self.computeLikelihoods(self.z_threshold,IndsA, IndsB, revLookUpA, revLookUpB, posAndHght[c],posHeightCov[c],poseTypeProb[c],
                    self.myTempTracks.posAndHeight,self.myTempTracks.posCov,self.myTempTracks.poseTypeProb)
                    
                # new work to do here... Evaluate all these combinations by re-inspecting the key points and analyze compatibility...
                for i2 in range(candInds.shape[0]):
                    iB = candInds[i2,1]
                    iA = candInds[i2,0]
                    for c2 in range(c):
                        # find earlier instances of this iB
                        iA2 = goodMatchesA[c2][np.where(goodMatchesB[c2] == iB)[0]]
                        if iA2.size == 1:
                            # now compare the two...
                            combWeight = keyPtsScrLoc[c][iA,self.poseHelper.indsForHeightEst] * keyPtsScrLoc[c2][iA2,self.poseHelper.indsForHeightEst]
                            hasCommonCat =  np.any(poseTypeProb[c][iA,:] * self.myTempTracks.poseTypeProb[iB,:] > 0.001)
                            err2, wght, l1Mod = self.assessPoseCompatibilities(channels[c],channels[c2],np.squeeze(unitVectorsLoc[c][iA,:,:]), \
                                np.squeeze(unitVectorsLoc[c2][iA2,:,:]),combWeight, hasCommonCat)
                            if l1Mod > -0.1:
                                if hasCommonCat:
                                    l1[i2] = l1[i2] * l1Mod / (l1[i2] * l1Mod + (1-l1[i2]) * (1-l1Mod)) # Bayesian update 
                                else:
                                    l1[i2] = l1Mod

                pTP = (1.0 - self.FA0) * np.power(Score[c][candInds[:,0]],0.25)
                l1 *=  pTP
                pFN = self.getFNprobs(channels[c], IndsB)
                l1 *= 1.0 - pFN[revLookUpB[candInds[:,1]]] 
                #doubleFalseProb = pFN[revLookUpB[candInds[:,1]]] * (1 - pTP)
                #l1 = l1 * (l1 > doubleFalseProb)
                pPost[revLookUpA[candInds[:,0]],revLookUpB[candInds[:,1]]] = l1 
            minRatio = 0.5
            moreMatchesA, moreMatchesB, toAdd, _, _, _, _ = graph_helper.solve_prob_graph(pPost,self.minProb,minRatio,
                0*self.minLogRatio, Score[c][IndsA], AisDistinct=isDistinct)   

            # make the updates for the matches
            for i2, i in enumerate(IndsA[moreMatchesA]):
                j = IndsB[moreMatchesB[i2]]
                self.myTempTracks.update_instance(j, timeStamps[c], posAndHght[c][i,:,:], posHeightCov[c][i,:,:,:], poseTypeProb[c][i,:], colorFeats[c][i,:], colorCov[c][i,:,:], hasColorA[c][i], channels[c], None, Orientations[c][i,:], Score[c][i])

            goodMatchesA[c] = IndsA[moreMatchesA]
            goodMatchesB[c] = IndsB[moreMatchesB]

            # add new items
            for i2, i in enumerate(IndsA[toAdd]): 
                goodMatchesA[c] = np.append(goodMatchesA[c],i)
                goodMatchesB[c] = np.append(goodMatchesB[c], self.myTempTracks.add_instance(posAndHght[c][i,:,:], posHeightCov[c][i,:,:,:], poseTypeProb[c][i,:], Score[c][i], timeStamps[c], colorFeats[c][i,:], colorCov[c][i,:,:], hasColorA[c][i], channels[c], None, Orientations[c][i,:], Score[c][i]))

        return goodMatchesA, goodMatchesB

    def track_humans(self, timeStamp, goodMatchesAin, goodMatchesBin):
        # match a consoldidated representation of frames @ one time point with the positions from earlier times (aka tracking)
        goodMatchesA = np.zeros(0,np.int)
        goodMatchesB = np.zeros(0,np.int)  
        IndsA = self.myTempTracks.Inds[self.myTempTracks.Score[self.myTempTracks.Inds]>self.det_threshold_add_new]
        if IndsA.size == 0:
            return goodMatchesA, goodMatchesB
        nA = IndsA.size
       
        IndsB = self.myTracks.Inds 
        revLookUpA = np.zeros(np.max(IndsA)+1,np.int)
        revLookUpB = np.zeros(self.myTracks.numAlloc, np.int)              

        revLookUpA[IndsA] = np.arange(IndsA.size)                    
        revLookUpB[IndsB] = np.arange(IndsB.size)
        toAdd = np.zeros(0, np.bool)
        # First: establish distinctness of the new data, i.e. are pairs of people mutually spatially resolved or not
        isDistinct = self.computeLikelihoods(self.distinct_z_threshold_tracking, IndsA, IndsA, revLookUpA, revLookUpA,
            self.myTempTracks.posAndHeight,self.myTempTracks.posCov,self.myTempTracks.poseTypeProb,
            self.myTempTracks.posAndHeight,self.myTempTracks.posCov,self.myTempTracks.poseTypeProb, AandBisSame=True)
        pPost = np.zeros((IndsA.size,IndsB.size), np.float)    
        if IndsB.size > 0:      
            l1, candInds = self.computeLikelihoods(self.z_threshold,IndsA, IndsB, revLookUpA, revLookUpB, self.myTempTracks.posAndHeight,\
                self.myTempTracks.posCov,self.myTempTracks.poseTypeProb,
                self.myTracks.posAndHeightTemp,self.myTracks.posCovTemp,self.myTracks.poseTypeProb,
                hasColorA=self.myTempTracks.colorFeatsValid(IndsA), colorFeatsA=self.myTempTracks.colorFeats, colorCovA=self.myTempTracks.colorCov, 
                hasColorB=self.myTracks.colorFeatsValid(IndsB), colorFeatsB=self.myTracks.colorFeats, colorCovB=self.myTracks.colorCov,
                OrientA=self.myTempTracks.bodyOrientation, OrientB=self.myTracks.bodyOrientation)
            pTP = (1.0 - self.FA0) * np.power(self.myTempTracks.Score[candInds[:,0]],0.2)

            l1 *=  pTP
            #pFN = self.getFNprobs(channels[c], IndsB)

            #l1 *= 1.0 - pFN[revLookUpB[candInds[:,1]]] 
            #doubleFalseProb = pFN[revLookUpB[candInds[:,1]]] * (1 - pTP)
            #l1 = l1 * (l1 > doubleFalseProb)
                
            pPost[revLookUpA[candInds[:,0]],revLookUpB[candInds[:,1]]] = l1   
                        
        minRatio = 0.5
        if np.max(self.myTracks.framesProcessed) < 2:
            minLogRatio = np.log(1.15)
        else:
            minLogRatio = self.minLogRatio
        moreMatchesA, moreMatchesB, toAdd, subGrphsA, subGrphsB, grphEdges, badSolA = \
            graph_helper.solve_prob_graph(pPost,self.minProb,minRatio,minLogRatio,
            self.myTempTracks.Score[IndsA], AisDistinct=isDistinct)   

        for g in range(len(subGrphsA)):
            if badSolA[g].size > 0:
                subsetsB = self.identifyLikelyDuplicates(grphEdges[g][badSolA[g],:], self.myTempTracks.posCov[IndsA[subGrphsA[g][badSolA[g]]],:,:])
                for k in range(len(subsetsB)):
                    self.myTracks.report_evidence_of_duplicates(IndsB[subGrphsB[g][subsetsB[k]]])

        for i2, i in enumerate(IndsA[moreMatchesA]):
            j = IndsB[moreMatchesB[i2]]
            self.myTracks.update_instance(j, timeStamp, self.myTempTracks.posAndHeight[i,:,:], \
                self.myTempTracks.posCov[i,:,:,:], self.myTempTracks.poseTypeProb[i,:], self.myTempTracks.colorFeats[i,:,:], \
                self.myTempTracks.colorCov[i,:,:], self.myTempTracks.colorFeatsValid(i), \
                None, self.myTempTracks.numObs[i,:], self.myTempTracks.bodyOrientation[i,:])


        goodMatchesA = np.append(goodMatchesA,IndsA[moreMatchesA])
        goodMatchesB = np.append(goodMatchesB,IndsB[moreMatchesB])

        # add new items
        for i2 in toAdd:
            i = IndsA[i2]  
            # add new item
            goodMatchesA = np.append(goodMatchesA,i)
            goodMatchesB = np.append(goodMatchesB, self.myTracks.add_instance(self.myTempTracks.posAndHeight[i,:,:], \
                self.myTempTracks.posCov[i,:,:,:], self.myTempTracks.poseTypeProb[i,:], self.myTempTracks.Score[i], \
                timeStamp, self.myTempTracks.colorFeats[i,:,:], self.myTempTracks.colorCov[i,:,:], \
                self.myTempTracks.colorFeatsValid(i), None, self.myTempTracks.numObs[i,:], self.myTempTracks.bodyOrientation[i,:]))

        # we ought to be done now except we need to do some maintenance on possible duplication
        # The philosophy is to be careful above to avoid merging falsely. NOw, below, we will allow evidence of duplication acculumate over multiple 
        # frames before taking any merging actions
        IndsB = self.myTracks.Inds
        self.myTracks.report_evidence_of_distinctness(goodMatchesB)

        wasRemoved, replacement = self.myTracks.detect_and_remove_duplicates()
        #print('{:d} entries removed.'.format(wasRemoved.size))
        
        # run through our return values in case some of them were affected by the duplicate removals
        swappers = np.where(np.isin(goodMatchesB,wasRemoved))[0]
        for i2, i in enumerate(swappers):
            replInd = np.where(wasRemoved==goodMatchesB[i])[0]
            goodMatchesB[swappers[i2]] = replacement[replInd[0]]

        # report on misses
        IndsB = self.myTracks.Inds
        misses = np.where(np.isin(IndsB,goodMatchesB,invert=True))[0]
        self.myTracks.reportOfMisses(-1, IndsB[misses])

        # Lastly:look for persons in the gallery who ought to get retired. 
        # we won't retire anyone seen in the present view, which is convenient as we won't have to revisit goodMatchesA&B
        minConseqMisses = np.min(self.myTracks.nConseqMisses[IndsB,:], axis=1)
        sumObs = np.sum(self.myTracks.numObs[IndsB,:], axis=1)
        removeThese = IndsB[np.where(np.logical_or(minConseqMisses > self.maxConseqMisses, minConseqMisses > sumObs))[0]]
        for j in removeThese:
            self.myTracks.removeHuman(j) 

        #Lastly, for real, make sure the references are right...
        matchesA = [None] * len(goodMatchesAin)
        matchesB = [None] * len(goodMatchesAin)
        for c in range(len(goodMatchesAin)):
            goodOnes = np.where(np.isin(goodMatchesBin[c],goodMatchesA))[0]
            if goodOnes.size > 0:
                matchesA[c] = goodMatchesAin[c][goodOnes]
                matchesB[c] = np.zeros(goodOnes.size,np.int)
                for i2, i in enumerate(goodOnes):
                    matchesB[c][i2] = goodMatchesB[np.where(goodMatchesA == goodMatchesBin[c][i])[0]]
            else:
                matchesA[c] = np.zeros(0,np.int)
                matchesB[c] = np.zeros(0,np.int)

        return matchesA, matchesB
    
    def assessPoseCompatibilities(self, chanA,chanB,uVecA,uVecB,combWeight, hasCommonCats):
        
        wghtThres = 0.5 
        minNumPts = 3
        # distance between two skew (non-intersecting) lines is https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d            
        
        d = self.myCamMappers[chanB].trans - self.myCamMappers[chanA].trans
        CPs = np.cross(uVecA,uVecB,axisa=0,axisb=0)
        Norms = np.linalg.norm(CPs,axis=-1)
        
        Dists = np.matmul(d,CPs.transpose()) / Norms
        combWeight *= np.abs(Norms > 0.025)
        #combWeight *= np.abs(Norms)

        numEls = max(minNumPts,np.sum((combWeight > wghtThres).astype(np.int)))

        sortInd = np.argsort(combWeight)[::-1]
        Inds = sortInd[0:numEls]
        
        totWeight = np.sum(combWeight[Inds])
        if totWeight < 1.2:
            return -1.0, totWeight, -1.0
        sqErr = np.sum(np.square(Dists[Inds]) * combWeight[Inds]) / totWeight
        if 0:
            # position of closest point http://www2.washjeff.edu/users/mwoltermann/Dorrie/69.pdf
            # doesn't seem to be helpful
            DPs = np.sum(uVecA*uVecB,axis=0)
            Dets = -1.0 + np.square(DPs)
            b1 = np.matmul(d,uVecA) 
            b2 = np.matmul(d,uVecB) 
            rs1 = (-b1 + DPs * b2) / Dets
            rs2 = (-DPs * b1 + b2) / Dets
            Points = 0.5 * (self.myCamMappers[chanA].trans + np.expand_dims(rs1,axis=1)*uVecA.transpose() + self.myCamMappers[chanB].trans + np.expand_dims(rs2,axis=1)*uVecB.transpose())
            meanPoint = np.sum(Points[Inds,0:2].transpose() * combWeight[Inds], axis=1) / totWeight

        # formula for likelihood ratio based on line intersections
        if hasCommonCats:
            Left = -6.3
            Right = -1.8
            Top = 0.92
            Bottom = 0.03
        else:
            Left = -7.0
            Right = 1.0
            Top = 0.95
            Bottom = 0.0
        logErr = np.log(sqErr)
        if logErr < Left:
            l1Mod = Top
        elif logErr > Right:
            l1Mod = Bottom
        else:
            l1Mod = (Right - logErr) * (Top - Bottom) / (Right - Left)

        return sqErr, totWeight, l1Mod
    
    def getVisitorMap(self, channelNum):
        return self.static_objects[channelNum].numVisits

    def get_static_objects(self, channelNum):
        return self.static_objects[channelNum].getStaticBoxes()

    def filterOutInconistentMatches2(self,channels, goodMatchesA, goodMatchesB, matchProbs, posAndHght,posHeightCov,poseTypeProb, unitVectorsLoc, keyPtsScrLoc):
        # catch inconsistencies between matched sightings in different frames, same time stamp
        for j in self.myTracks.Inds:
            ch = np.zeros(0,np.int)
            Ind = np.zeros(0,np.int)
            for c in range(len(goodMatchesA)):      
                matchInd = np.where(goodMatchesB[c] == j)[0]
                if matchInd.size > 0:
                    ch = np.append(ch,c)
                    Ind = np.append(Ind,matchInd)
            if ch.size > 1:
                # check consistency now
                numMatches = ch.size
                zVals = np.zeros((numMatches,numMatches))
                l1 = np.ones((numMatches,numMatches))

                locMatchProbs = np.zeros(numMatches)
                includeIt = np.ones(numMatches, np.bool)

                for k1 in range(numMatches):
                    c1 = ch[k1]
                    i1 = goodMatchesA[c1][Ind[k1]]
                    locMatchProbs[k1] = matchProbs[c1][Ind[k1]]
                    for k2 in range(k1):
                        # compute the spatial prob                        
                        c2 = ch[k2]
                        i2 = goodMatchesA[c2][Ind[k2]]
                        # now compare the two...
                        Diff = np.array([posAndHght[c1][i1,-1,:]-posAndHght[c2][i2,-1,:]])
                        combCov = posHeightCov[c1][i1,-1,:,:]+posHeightCov[c2][i2,-1,:,:]
                        zVals[k1,k2] = np.matmul(Diff,np.matmul(np.linalg.inv(combCov),Diff.transpose()))
                        l1[k1,k2] = 1.0 - self.myChiDistGuy.cdf(zVals[k1,k2], np.array(3,np.int))                        
                        combWeight = keyPtsScrLoc[c1][i1,self.poseHelper.indsForHeightEst] * keyPtsScrLoc[c2][i2,self.poseHelper.indsForHeightEst]
                        hasCommonCat =  np.any(poseTypeProb[c1][i1,:] * poseTypeProb[c2][i2,:] > 0.001)
                        err2, wght, l1Mod = self.assessPoseCompatibilities(channels[c1],channels[c2],np.squeeze(unitVectorsLoc[c1][i1,:,:]), \
                            np.squeeze(unitVectorsLoc[c2][i2,:,:]),combWeight, hasCommonCat)
                        if l1Mod > -0.1:
                            if hasCommonCat:
                                l1[k1,k2] = l1[k1,k2] * l1Mod / (l1[k1,k2] * l1Mod + (1-l1[k1,k2]) * (1-l1Mod)) # Bayesian update 
                            else:
                                l1[k1,k2] = l1Mod
                        elif not hasCommonCat:
                                l1[k1,k2] = 0.0

                for k in np.argsort(locMatchProbs):
                    consistencyThreshold = 0.07
                    numIncons = np.sum(l1[k,:] < consistencyThreshold) + np.sum(l1[:,k] < consistencyThreshold)     
                    if numIncons > 0:
                        includeIt[k] = False
                        l1[k,:] = 1.0
                        l1[:,k] = 1.0

                for k in np.where(includeIt == False)[0]:
                    goodMatchesA[ch[k]] = np.delete(goodMatchesA[ch[k]],Ind[k])
                    goodMatchesB[ch[k]] = np.delete(goodMatchesB[ch[k]],Ind[k])


        return goodMatchesA, goodMatchesB  

    def filterOutInconistentMatches(self,goodMatchesA, goodMatchesB, matchProbs, posAndHght,posHeightCov,poseTypeProb):
        # catch inconsistencies between matched sightings in different frames, same time stamp
        for j in self.myTracks.Inds:
            ch = np.zeros(0,np.int)
            Ind = np.zeros(0,np.int)
            for c in range(len(goodMatchesA)):      
                matchInd = np.where(goodMatchesB[c] == j)[0]
                if matchInd.size > 0:
                    ch = np.append(ch,c)
                    Ind = np.append(Ind,matchInd)
            if ch.size > 1:
                # check consistency now
                numMatches = ch.size
                zVals = np.zeros((numMatches,numMatches))
                locMatchProbs = np.zeros(numMatches)
                includeIt = np.ones(numMatches, np.bool)
                #numInconsistencies = np.zeros(numMatches, np.int)

                for k1 in range(numMatches):
                    c1 = ch[k1]
                    i1 = goodMatchesA[c1][Ind[k1]]
                    locMatchProbs[k1] = matchProbs[c1][Ind[k1]]
                    for k2 in range(k1):
                        # compute the spatial prob                        
                        c2 = ch[k2]
                        i2 = goodMatchesA[c2][Ind[k2]]
                        Diff = np.array([posAndHght[c1][i1,-1,:]-posAndHght[c2][i2,-1,:]])
                        combCov = posHeightCov[c1][i1,-1,:,:]+posHeightCov[c2][i2,-1,:,:]
                        zVals[k1,k2] = np.matmul(Diff,np.matmul(np.linalg.inv(combCov),Diff.transpose()))
                        if 0:
                            if zVals[k1,k2] > self.z_consist_threshold:
                                if locMatchProbs[k1] > locMatchProbs[k2]:
                                    includeIt[k2] = False
                                    numInconsistencies[k2] += 1
                                else:
                                    includeIt[k1] = False
                                    numInconsistencies[k1] += 1

                for k in np.argsort(locMatchProbs):
                    numIncons = np.sum(zVals[k,:] > self.z_consist_threshold) + np.sum(zVals[:,k] > self.z_consist_threshold)
                    if numIncons > 0:
                        includeIt[k] = False
                        zVals[k,:] = 0.0
                        zVals[:,k] = 0.0

                for k in np.where(includeIt == False)[0]:
                    goodMatchesA[ch[k]] = np.delete(goodMatchesA[ch[k]],Ind[k])
                    goodMatchesB[ch[k]] = np.delete(goodMatchesB[ch[k]],Ind[k])


        return goodMatchesA, goodMatchesB  

    def computeLikelihoods2(self,zThres,IndsA, IndsB, chA, revLookUpB, posAndHghtA,posCovA, poseProbA, posAndHghtB,posCovB, poseProbB,
        hasColorA=None, colorFeatsA=None, colorCovA=None, hasColorB=None, colorFeatsB=None, colorCovB=None, OrientA=None, OrientB=None,
        AandBisSame=False):         
        goodIndsA = np.argwhere(poseProbA[IndsA,:] > 0.01)
        goodIndsB = np.argwhere(poseProbB[IndsB,:] > 0.01)
        goodIndsA[:,0] = IndsA[goodIndsA[:,0]]
        goodIndsB[:,0] = IndsB[goodIndsB[:,0]]

        muXdiff = posAndHghtB[goodIndsB[:,0],goodIndsB[:,1],0] - np.expand_dims(posAndHghtA[goodIndsA[:,0],goodIndsA[:,1],0],axis=1)
        muYdiff = posAndHghtB[goodIndsB[:,0],goodIndsB[:,1],1] - np.expand_dims(posAndHghtA[goodIndsA[:,0],goodIndsA[:,1],1],axis=1)
        
        totDistSq = np.square(muXdiff) + np.square(muYdiff)

        posVarA = np.expand_dims(posCovA[goodIndsA[:,0],goodIndsA[:,1],0,0] + posCovA[goodIndsA[:,0],goodIndsA[:,1],1,1], axis=1) 

        candInds2 = np.argwhere(np.logical_or(totDistSq < np.square(4.0), totDistSq - 4 * posVarA < 0.0))
        candInds = np.zeros(candInds2.shape, np.int)      
  
        if candInds.shape[0] > 0:
            candInds[:,0] = goodIndsA[:,0][candInds2[:,0]]
            candInds[:,1] = goodIndsB[:,0][candInds2[:,1]]
            _, ic = np.unique(candInds,axis=0,return_index=True)
            candInds = candInds[ic,:]
            if AandBisSame:
                # this means we're comparing a set to itself. So we can skip the upper triangle of the matrix
                # we have to be careful though since the quick metric used above to filter is not symmetric in A & B
                isDistinct = np.ones((IndsA.size,IndsA.size),np.bool)
                isDistinct[revLookUpA[candInds[:,0]],revLookUpA[candInds[:,1]]] = False
                isDistinct = np.logical_and(isDistinct,isDistinct.transpose())
                isDistinct[np.arange(IndsA.size),np.arange(IndsA.size)]= True
                candInds = IndsA[np.argwhere(isDistinct==False)]
                lowTriInds = np.where(candInds[:,0] < candInds[:,1])[0]
                candInds = candInds[lowTriInds,:]

        NCands = candInds.shape[0]
        zValue = np.zeros(NCands, np.float)
        DoF = 3 * np.ones(NCands, np.int)
        areaFactors = np.ones(NCands, np.float)

        for kk in range(NCands):
            i = candInds[kk,0]
            j = candInds[kk,1]

            probProd = poseProbA[i,:] * poseProbB[j,:]
            commonInds = np.where(probProd > 0.0001)[0]

            if commonInds.size == 0:
                indPair = np.array([[np.argmax(poseProbA[i,:])],[np.argmax(poseProbB[j,:])]])
            else:
                indPair = np.vstack((commonInds,commonInds))
            bestL = 0.0
            zValue[kk] = 1E16
            bestZratio = 1E30
            for qq in range(indPair.shape[1]):                 
                Diff = np.array([posAndHghtB[j,indPair[1,qq],:]-posAndHghtA[i,indPair[0,qq],:]])
                combCov = posCovB[j,indPair[1,qq],:,:]+posCovA[i,indPair[0,qq],:,:]
                zVal = np.matmul(Diff,np.matmul(np.linalg.inv(combCov),Diff.transpose()))
                zVal2D = np.matmul(Diff[:,0:2],np.matmul(np.linalg.inv(combCov[0:2,0:2]),Diff[:,0:2].transpose()))
                if zVal2D < 0.6 * zVal:
                    zVal = zVal2D
                    DoF[kk] = 2
                if AandBisSame:
                    distinct = zVal > zThres
                    isDistinct[revLookUpA[i],revLookUpB[j]] = distinct
                    isDistinct[revLookUpB[j],revLookUpA[i]] = distinct                        
                else:
                    if zVal < zThres:
                        personalSpaceOfManSqMeters = 1.#1.75 
                        # compute the area of the error ellipse (basically, if computing the eigenvalues and the area of 
                        # the ellipse defined by it, we get the answer below
                        areaOfErrEllipse = np.pi * np.sqrt(combCov[0,0] * combCov[1,1] - combCov[1,0] * combCov[0,1])
                        areaFactor = personalSpaceOfManSqMeters / max(personalSpaceOfManSqMeters,areaOfErrEllipse)

                        zRatio = zVal / areaFactor
                        if zRatio < bestZratio:
                            bestZratio = zRatio
                            zValue[kk] = zVal
                            areaFactors[kk] = areaFactor
    
        newCandInds = np.where(zValue < zThres)[0]
        NCands = newCandInds.size
        candInds = candInds[newCandInds,:]
        if 1:
            l1 = areaFactors[newCandInds] * (1.0 - self.myChiDistGuy.cdf(zValue[newCandInds], DoF[newCandInds]))
        else:
            l1 = areaFactors[newCandInds] * (1.0 - self.myChiDistGuy.cdf(zValue[newCandInds], 3*np.ones(NCands,np.int)))

        l1Spatial = l1.copy()
        if 1 and not hasColorA is None:
            KL = np.zeros((NCands,self.numChannels), np.float)
            P1ToP0col = np.ones((NCands,self.numChannels), np.float) # likelihood ratio of beign different vs same person

            # http://math.bme.hu/~marib/tobbvalt/tv5.pdf
            # now look for cases where is a need for color info in order to decide...
            for kk in range(NCands):
                i = candInds[kk,0]
                j = candInds[kk,1]
                c = chA
                if hasColorA[i] and hasColorB[revLookUpB[j],c]:
                    Cov1 = colorCovB[j,c,:,:]
                    Cov0 = colorCovA[i,:,:]

                    if 1 :
                        Diff = np.array([colorFeatsA[i,:] - colorFeatsB[j,c,:]])
                        invCov = np.linalg.inv(Cov1 + Cov0)
                        zVal = np.matmul(Diff,np.matmul(invCov,Diff.transpose()))
                        if 0:
                            l2 = 1.0 - self.myChiDistGuy.cdf(zVal[0,:], np.array(self.numColorFeats))
                            l1New = l1[kk] * l2
                            l1NewNot = (1-l1[kk]) * (1-l2)
                            l1[kk] = l1New #/ (l1New + l1NewNot)
                        else:
                            l1[kk] = areaFactors[newCandInds[kk]] * (1.0 - self.myChiDistGuy.cdf(np.array(zVal[0,0]+zValue[newCandInds[kk]]), np.array(self.numColorFeats+DoF[newCandInds[kk]])))

                    else:
                        det1 = np.linalg.det(Cov1)
                        det0 = np.linalg.det(Cov0)
                        if det1 > 1E-16 and det0 > 1E-16: 
                            Diff = np.array([colorFeatsA[i,:] - colorFeatsB[j,c,:]])
                            invCov1 = np.linalg.inv(Cov1)
                            # compute KL divergence
                            KL[kk,c] = 0.5 * (np.trace(np.matmul(invCov1,Cov0)) + np.matmul(Diff, np.matmul(invCov1, Diff.transpose())) - self.numColorFeats + np.log(det1/det0))   
                            #P1ToP0col[kk,c] = np.interp(KL[kk,c],self.colPrbFn[c,:,0],self.colPrbFn[c,:,1],right=self.colPrbFn[c,-1,1])
                            P1ToP0col[kk,c] = np.interp(KL[kk,c],self.colPrbFn[:,0],self.colPrbFn[:,1],right=self.colPrbFn[-1,1])
                            l1[kk] /= l1[kk]  + (1 - l1[kk]) * P1ToP0col[kk,c] # a Bayesian update for the probability of matching...
                if self.doOrientation:
                    # let's do orientation too!
                    lkRatioIsDifferent = self.compareOrientions(OrientA[i,:], OrientB[j,:])
                    l1[kk] /= l1[kk]  + (1 - l1[kk]) * lkRatioIsDifferent
            d = 1
        if AandBisSame:
            return isDistinct
        return l1, candInds

    def match_humans_in_frame(self, channels, goodInds, timeStamps, posAndHght, posHeightCov, poseTypeProb, Score, colorFeats,
        colorCov, hasColorA, Orientations, unitVectorsLoc, keyPtsScrLoc):
        N = channels.shape[0]
        
        numPeopleAtStart = self.myTracks.Inds.size
        nA = np.zeros(N,np.int)
        goodMatchesA = [None] * N
        goodMatchesB = [None] * N
        matchProbs = [None] * N
        toBeAdded = [None] * N
        toBeAddedB = [None] * N 
        hasColorB = [None] * N 
        isDistinct = [None] * N 

        for c in range(N):
            nA[c] = Score[c].shape[0]
            goodMatchesA[c] = np.zeros(0,np.int)
            goodMatchesB[c] = np.zeros(0,np.int)   
            matchProbs[c] = np.zeros(0)  
        
        revLookUpA = np.zeros(np.max(nA)+1,np.int)
        revLookUpB = np.zeros(self.myTracks.numAlloc, np.int)
        zeroProb = np.min(self.minProb) / 2 
        zeroProbPlus = zeroProb + 0.00001
        minProbLog = np.log(self.minProb)

        numIts = 2
        for it in range(numIts):
            # in these loops we can match to gallery, and do an update inline, and we can also add to gallery.
            # we can't delete or merge though. that needs to happen afterwards            
            for c in range(N):                  
                IndsB = self.myTracks.Inds[np.isin(self.myTracks.Inds, goodMatchesB[c], invert=True)]  # get all gallery folks that haven't already been matched in frame
                if self.myTracks.numAlloc > revLookUpB.size:
                    revLookUpB.resize(self.myTracks.numAlloc)
                hasColorB = self.myTracks.colorFeatsValid(IndsB) 
                IndsA = goodInds[c][np.where(np.isin(goodInds[c], goodMatchesA[c], invert=True))[0]]
                if IndsA.size == 0:
                    continue
                revLookUpA[IndsA] = np.arange(IndsA.size)     
                if 1 or it == 0:
                    isDistinct[c] = self.computeLikelihoods(self.distinct_z_threshold_intra_frame, IndsA, IndsA, revLookUpA, revLookUpA, posAndHght[c],posHeightCov[c],
                        poseTypeProb[c], posAndHght[c],posHeightCov[c],poseTypeProb[c],AandBisSame=True)
                revLookUpB[IndsB] = np.arange(IndsB.size)
                toAdd = np.zeros(0, np.bool)
                pPost = np.zeros((IndsA.size,IndsB.size), np.float)  
                self.FA0 = 0.0015 
                pTP = (1.0 - self.FA0) * np.power(Score[c][IndsA],0.2)
                pFN = self.getFNprobs(channels[c], IndsB)
                if IndsB.size > 0:    
                    l1, candInds = self.computeLikelihoods2(self.z_threshold,IndsA, IndsB, c, revLookUpB, posAndHght[c],posHeightCov[c],poseTypeProb[c],
                        self.myTracks.posAndHeightTemp,self.myTracks.posCovTemp,self.myTracks.poseTypeProb,
                        hasColorA[c], colorFeats[c], colorCov[c], hasColorB, self.myTracks.colorFeats, self.myTracks.colorCov,
                        Orientations[c], self.myTracks.bodyOrientation)                                                                               
                    pPost[revLookUpA[candInds[:,0]],revLookUpB[candInds[:,1]]] = l1   
                        
                minRatio = 0.5
                if np.max(self.myTracks.framesProcessed) < 2:
                    minLogRatio = np.log(1.15) # because there is so much uncertainty in the second frame
                else:
                    minLogRatio = self.minLogRatios[it] #self.minLogRatio
                moreMatchesA, moreMatchesB, toAdd, subGrphsA, subGrphsB, grphEdges, badSolA = \
                    graph_helper.solve_prob_graph2(pPost, 1.0-pTP, pFN, minRatio, minLogRatio,
                    Score[c][IndsA], AisDistinct=isDistinct[c])   

                for g in range(len(subGrphsA)):
                    if badSolA[g].size > 0:
                        subsetsB = self.identifyLikelyDuplicates(grphEdges[g][badSolA[g],:], posHeightCov[c][IndsA[subGrphsA[g][badSolA[g]]],:,:,:])
                        for k in range(len(subsetsB)):
                            self.myTracks.report_evidence_of_duplicates(IndsB[subGrphsB[g][subsetsB[k]]])

                if (not self.enforceConsistency) or it > 0:
                    # make the updates for the matches
                    for i2, i in enumerate(IndsA[moreMatchesA]):
                        j = IndsB[moreMatchesB[i2]]
                        self.myTracks.update_instance_kalman(j, timeStamps[c], posAndHght[c][i,:,:], posHeightCov[c][i,:,:,:], poseTypeProb[c][i,:], colorFeats[c][i,:], colorCov[c][i,:,:], hasColorA[c][i], channels[c],None, Orientations[c][i,:])

                goodMatchesA[c] = np.append(goodMatchesA[c],IndsA[moreMatchesA])
                goodMatchesB[c] = np.append(goodMatchesB[c],IndsB[moreMatchesB])
                if IndsB.size > 0 and it == 0:
                    matchProbs[c] = np.append(matchProbs[c],pPost[moreMatchesA,moreMatchesB])

                # add new items
                for i2 in toAdd:
                    i = IndsA[i2]  
                    if Score[c][i] > self.det_threshold_add_new:
                        # add new item
                        goodMatchesA[c] = np.append(goodMatchesA[c],i)
                        goodMatchesB[c] = np.append(goodMatchesB[c], self.myTracks.add_instance(posAndHght[c][i,:,:], posHeightCov[c][i,:,:,:], poseTypeProb[c][i,:], Score[c][i], timeStamps[c], colorFeats[c][i,:], colorCov[c][i,:,:], hasColorA[c][i], channels[c],None, Orientations[c][i,:]))
                        if it == 0 and self.enforceConsistency:
                            matchProbs[c] = np.append(matchProbs[c],1.0)

            if self.enforceConsistency and it == 0:
                # now update all the matches, but first, check for consistency between them...
                goodMatchesA, goodMatchesB = self.filterOutInconistentMatches2(channels, goodMatchesA, goodMatchesB, matchProbs, posAndHght,posHeightCov,poseTypeProb, unitVectorsLoc, keyPtsScrLoc)
                for c in range(N):  
                    for i2, i in enumerate(goodMatchesA[c]):
                        if matchProbs[c][i2] < 1.0:
                            j = goodMatchesB[c][i2]
                            self.myTracks.update_instance_kalman(j, timeStamps[c], posAndHght[c][i,:,:], posHeightCov[c][i,:,:,:], poseTypeProb[c][i,:], colorFeats[c][i,:], colorCov[c][i,:,:], hasColorA[c][i], channels[c], None, Orientations[c][i,:])


        # we ought to be done now except we need to do some maintenance on possible duplication
        # The philosophy is to be careful above to avoid merging falsely. NOw, below, we will allow evidence of duplication acculumate over multiple 
        # frames before taking any merging actions
        IndsB = self.myTracks.Inds
        for c in range(N):
            self.myTracks.report_evidence_of_distinctness(goodMatchesB[c])

        wasRemoved, replacement = self.myTracks.detect_and_remove_duplicates()
        #print('{:d} entries removed.'.format(wasRemoved.size))
        
        # run through our return values in case some of them were affected by the duplicate removals
        for c in range(N):
            swappers = np.where(np.isin(goodMatchesB[c],wasRemoved))[0]
            for i2, i in enumerate(swappers):
                replInd = np.where(wasRemoved==goodMatchesB[c][i])[0]
                goodMatchesB[c][swappers[i2]] = replacement[replInd[0]]
        
        # report on misses
        IndsB = self.myTracks.Inds
        for c in range(N):
            misses = np.where(np.isin(IndsB,goodMatchesB[c],invert=True))[0]
            self.myTracks.reportOfMisses(channels[c], IndsB[misses])

        # Lastly:look for persons in the gallery who ought to get retired. 
        # we won't retire anyone seen in the present view, which is convenient as we won't have to revisit goodMatchesA&B
        minConseqMisses = np.min(self.myTracks.nConseqMisses[IndsB,:], axis=1)
        sumObs = np.sum(self.myTracks.numObs[IndsB,:], axis=1)
        removeThese = IndsB[np.where(np.logical_or(minConseqMisses > self.maxConseqMisses, minConseqMisses > sumObs))[0]]
        for j in removeThese:
            self.myTracks.removeHuman(j) 
        return goodMatchesA, goodMatchesB

    def report_frame(self,detResults, bgTrackers = None):
        if self.doFaceID:
            self.check_on_face_workers()

        freshChannels = np.zeros(0,np.int)
        for c in range(self.numChannels):
            if detResults[c]["detectionWasRun"]:
                freshChannels = np.append(freshChannels,c)
        startTime = time.time()
        # now, done all with the updates, we just need to package the appropriate information back to our caller
        # for now, let's just return the identity for each human that was identified or added
        trackingResults = {"Description": [None] * self.numChannels,
            "goodMatches": [None] * self.numChannels,
            "freshChannels": freshChannels
            }
        N = freshChannels.shape[0]
        
        if N > 0:
            goodInds = [None] * N
            colorFeats = [None] * N
            colorCov = [None] * N
            hasColor = [None] * N
            posAndHghtLoc = [None] * N 
            unitVectorsLoc = [None] * N 
            keyPtsScrLoc = [None] * N 
            posHeightCovLoc = [None] * N 
            poseTypeProbLoc = [None] * N 
            Scores = [None] * N 
            OrientationsLoc = [None] * N 
            timeStamps = np.zeros(N,np.float)

            for i2, c in enumerate(freshChannels):
                Res = detResults[c]
                if self.detectStaticObjects:
                    Res["includeIt"] = np.logical_and(Res["includeIt"], self.static_objects[c].is_static_object(Res["keypoint_coords"], Res["keypoint_scores"],np.arange(0,Res["num"])) == -1)
                goodInds[i2] = np.where(Res["includeIt"])[0]
                posAndHghtLoc[i2] = Res["posAndHght"]
                posHeightCovLoc[i2] = Res["posCov"]
                poseTypeProbLoc[i2] = Res["poseTypeProb"]
                Scores[i2] = Res["scores"]
                OrientationsLoc[i2] = Res["Orientations"]
                colorFeats[i2] = Res["colorFeats"]
                colorCov[i2] = Res["colorCov"]
                hasColor[i2] = Res["hasColor"]
                for i in range(hasColor[i2].size):
                    if hasColor[i2][i]:
                        hasColor[i2][i] = self.poseHelper.getNumPixelsTorso(Res["keypoint_coords"][i,:,:]) > 0*500.0
                timeStamps[i2] = Res["timeStamp"]
                unitVectorsLoc[i2] = Res["unitVectors"]
                keyPtsScrLoc[i2] = Res["keypoint_scores"]
                           
            self.myTracks.predict_kalman(np.mean(timeStamps))
            
            if self.oldAlgo:
                goodMatchesA, goodMatchesB = self.match_humans_in_frame(freshChannels, goodInds, timeStamps, posAndHghtLoc, posHeightCovLoc, poseTypeProbLoc, Scores, colorFeats, colorCov, hasColor, OrientationsLoc, unitVectorsLoc, keyPtsScrLoc)
            else:
                goodMatchesA, goodMatchesB = self.match_humans_in_frame_only(freshChannels, goodInds, timeStamps, posAndHghtLoc, posHeightCovLoc, poseTypeProbLoc, Scores, colorFeats, colorCov, hasColor, OrientationsLoc, unitVectorsLoc, keyPtsScrLoc)                
                goodMatchesA, goodMatchesB = self.track_humans(np.mean(timeStamps),goodMatchesA, goodMatchesB)                             

            for i2, c in enumerate(freshChannels):
                Res = detResults[c]
                trackingResults["Description"][c] = []  
                trackingResults["goodMatches"][c] = np.zeros(0,np.int)
            
                for j2,j in enumerate(goodMatchesB[i2]):
                    thisI = goodMatchesA[i2][j2]
                    if self.myTracks.numObs[j,c] >= self.minNumSightings:  
                        trackingResults["Description"][c].append(self.myTracks.makeCaption(j)) 
                        trackingResults["goodMatches"][c] = np.append(trackingResults["goodMatches"][c], thisI)
                    if self.doFaceID and Res["frameNum"] % self.faceIDinterval == 0:
                        # will look for a face, providing the person is not moving away from the camera
                        if np.sum(self.myTracks.numObs[j,:]) > 3 and Res["faceOpportunity"][thisI]:
                            face_image = Res["faceChip"][thisI]    
                            
                            if face_image.shape[0] < 58:
                                continue

                            if self.myTracks.db_ids[j] == None:
                                request_content = {
                                    'type': 'identify',
                                    'image' : face_image,
                                    'temp_id' : self.myTracks.temp_ids[j]}

                                self.face_input_queue.put(RequestMessage(request_content, j, self.myTracks.numReincarnations[j]))
                                
                                #cv2.imwrite('face'+str(self.total_face_jobs)+'.png',face_image)
                                
                                self.total_face_jobs += 1

                            elif 0:
                                # asks for verification of ID. Not needed right now, but maybe later
                                request_content = {
                                    'type': 'confirm_and_update',
                                    'image' : face_image,
                                    'db_id' : self.myTracks.db_ids[j]}
                                self.face_pool.submit_request(request_content, j, self.myTracks.numReincarnations[j])
       
                if self.detectStaticObjects:
                    self.static_objects[c].updateActivityScene(Res["boxes"],goodMatchesA[i2])
                    Buff = np.ones(Res["num"],np.bool)
                    Buff[Scores[i2] > 0.55] = False
                    self.static_objects[c].update_static_objects(Res["keypoint_coords"],Res["keypoint_scores"],np.where(Buff)[0])
    
                self.myTracks.frameProcessed(c)

        if self.includePosInfo:
            indsToReport = self.myTracks.Inds[np.where(np.sum(self.myTracks.nConseqHits[self.myTracks.Inds,:], axis=1)>0)[0]]
            trackingResults["Pos"] = self.myTracks.posAndHeightTemp[indsToReport,-1,0:3]
            trackingResults["Cov"] = self.myTracks.posCovTemp[indsToReport,-1,0:2,0:2]
            trackingResults["Inds"] = indsToReport
            trackingResults["numReincarnations"] = self.myTracks.numReincarnations[indsToReport]
        if self.exportStaticObjects:
            trackingResults["staticObjects"] = [None] * self.numChannels 
            for c in range(self.numChannels): 
                trackingResults["staticObjects"][c] = self.get_static_objects(c)

        if self.exportVisitorMap:
            trackingResults["visitorMap"] = [None] * self.numChannels 
            for c in range(self.numChannels):                    
                trackingResults["visitorMap"][c] = self.getVisitorMap(c)       
             
        if not bgTrackers is None:
            # see if there are left packages...
            trackingResults["leftLuggageBBs"] = [None] * self.numChannels
            trackingResults["leftLuggageInds"] = [None] * self.numChannels
            trackingResults["leftLuggageOwnerID"] = np.zeros((0,2),np.int)
            leftLugggagePos = np.zeros((0,2))
            timeAppeared = np.zeros(0)
            numLuggObs = np.zeros(0)
            for c in range(self.numChannels):
                stableOnes = np.where(bgTrackers[c].BoxesStable)[0]
                trackingResults["leftLuggageBBs"][c] = bgTrackers[c].Boxes[stableOnes]
                trackingResults["leftLuggageInds"][c] = np.zeros(stableOnes.size,np.int)   #np.zeros(bgTrackers[c].Boxes.shape[0],np.int) 
                for i2, i in enumerate(stableOnes): #range(bgTrackers[c].Boxes.shape[0]):          
                    thisPos = self.myCamMappers[c].getGroundPosition(int(0.5*(bgTrackers[c].Boxes[i,1]+bgTrackers[c].Boxes[i,3])),
                        int(bgTrackers[c].Boxes[i,2]-0.3*(bgTrackers[c].Boxes[i,2]-bgTrackers[c].Boxes[i,0])) ) 
                    thisPos = np.array((thisPos[0][0],thisPos[1][0]))
                    addNew = True
                    if leftLugggagePos.shape[0] > 0:
                        # compare to previous entries
                        Dists = np.linalg.norm(leftLugggagePos - thisPos, axis = 1)
                        closest = np.argmin(Dists)
                        if Dists[closest] < 1.5:
                            # merge with pre-existing
                            trackingResults["leftLuggageInds"][c][i2] = closest
                            timeAppeared[closest] = min(timeAppeared[closest], bgTrackers[c].BoxesTimeAppeared[i]) 
                            leftLugggagePos[closest,:] = (thisPos + numLuggObs[closest] * leftLugggagePos[closest,:]) / (numLuggObs[closest] + 1)
                            numLuggObs[closest] += 1
                            addNew = False
                    if addNew:
                        trackingResults["leftLuggageInds"][c][i2] = leftLugggagePos.shape[0]
                        leftLugggagePos = np.vstack((leftLugggagePos, thisPos))
                        numLuggObs = np.append(numLuggObs,1)
                        timeAppeared = np.append(timeAppeared, bgTrackers[c].BoxesTimeAppeared[i]) 
            for i in range(numLuggObs.size):
                compID = self.myTracks.whoWasThere(leftLugggagePos[i,:], timeAppeared[i])
                if compID.size == 2: # means one match
                    trackingResults["leftLuggageOwnerID"] = np.vstack((trackingResults["leftLuggageOwnerID"],compID))
                else:
                    trackingResults["leftLuggageOwnerID"] = np.vstack((trackingResults["leftLuggageOwnerID"],np.array((-1,-1))))

        trackingResults["trackTime"] = time.time() - startTime
        return trackingResults

    def check_on_face_workers(self):
        while not self.face_output_queue.empty():
            worker_response = self.face_output_queue.get()
            print('response time ', worker_response.response_time)
            with open(self.log_path, 'a+') as log:
                log.write('%d--%f\n' %(time.time(), worker_response.response_time))
            self.total_response_time += worker_response.response_time
            if worker_response.response['status'] == 'ok':
                self.successful_face_reqs += 1
            else:
                self.unsuccessful_face_reqs += 1

            if self.myTracks.numReincarnations[worker_response.id] == worker_response.reincarns:
                self.myTracks.face_detected[worker_response.id] = True
                if 'id' in worker_response.response:
                    self.myTracks.db_ids[worker_response.id] = worker_response.response['id']
                    self.myTracks.names[worker_response.id] = worker_response.response['name']
            else:
                self.myTracks.face_detected[worker_response.id] = False


    def how_far_they_go(self, ID, elapsedTime):
        return self.myTracks.velo2D[ID,:] * elapsedTime
                 

def trackerWorker(track_in_q,track_out_q,track_comm_q,channels, trckCfg, face_input_q, face_output_q):
    mySceneGuy = scene_manager(channels, trckCfg, face_input_q, face_output_q)
    # need to send a copy of the camMappers back to the mother ship
    track_comm_q.put(mySceneGuy.myCamMappers)
    numChannels = len(channels)
    detResults = [None] * numChannels 

    while True:       
        for c in range(numChannels):
            detResults[c] = track_in_q[c].get()
            if len(detResults[c]) == 3:
                if detResults[c] == "die":
                    mySceneGuy.shutdown()
                    return
        if detResults[c]["frameNum"] > -1:
            track_out_q.put(mySceneGuy.report_frame(detResults))

class sceneMgr:
    def __init__(self, channels, trckCfg, face_input_q=None, face_output_q=None): 

        self.track_in_q = [] 
        self.track_comm_q = Queue()
        for c in range(len(channels)):
            self.track_in_q.append(Queue())
        self.track_out_q = Queue()
        self.track_pool = multiprocessing.Pool(1, trackerWorker, (self.track_in_q,self.track_out_q,self.track_comm_q,channels, trckCfg, face_input_q, face_output_q))  
        self.myCamMappers = self.track_comm_q.get()

    def shutdown(self):
        for c in range(len(self.track_in_q)):
            self.track_in_q[c].put("die")

    def report_frame(self, detResults):
        # this function just makes the interface the same whether tracker runs in sep thread or not
        return self.track_out_q.get()

def enforceConfigCoherence(trckCfg, video_channels, myChannels, face_in_q=None, face_out_q=None):
     # enforcing some configuration constraints 
    trckCfg["liveCapture"] = not myChannels[0].useRecordedVideo
    if trckCfg["recordData"]:
        trckCfg["runLiveGPUdetection"] = False
        trckCfg["doRendering"] = False
    if trckCfg["liveCapture"]:
        trckCfg["cacheDetResults"] = False # will only allow caching det results if it is a video
        trckCfg["dropFramesWhenBehind"] = True # If we run live, we need to keep up with the dataflow    
        if not trckCfg["recordData"]:
            # we're running live cameras and not recording. Better run the detection & rendering then, if not, why are we doing this?
            trckCfg["runLiveGPUdetection"] = True
            trckCfg["doRendering"] = True
  
    if trckCfg["recordData"]:
        # need to make sure we have a shared name for the recording between all the channels
        trckCfg["rec_name"] = video_channels.get_recording_name(myChannels[0])        
        myTracker = []
    else:
        print("Setting up channels...")
        if trckCfg["runTrackerInSepProcess"]:
            sceneType = sceneMgr
        else:
            sceneType = scene_manager           
        myTracker = sceneType(myChannels, trckCfg, face_in_q, face_out_q)  

    return trckCfg, myTracker

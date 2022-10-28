from __future__ import print_function
import numpy as np
import scipy.io as sio
import pickle
import scipy.stats
import matplotlib.pyplot as plt

useCachedResults = False
analyzeIt = False

if useCachedResults:      
    f = open("intersect_error_data.npz", "rb")
    savedErrors = pickle.load(f)
    commonCats = pickle.load(f)
    savedLabels = pickle.load(f)
    savedL1 = pickle.load(f)
    savedL1spatial = pickle.load(f)
    visibility = pickle.load(f)
    percInFrame = pickle.load(f)
    vertSizeFractions = pickle.load(f)
    Detected = pickle.load(f)
    FN0 = pickle.load(f)
    vertSizeFractionsFA = pickle.load(f)
    ScoreFA = pickle.load(f)
    LabelFA = pickle.load(f)
    #orFeats = pickle.load(f)
    savedKL = pickle.load(f)
    colorLabel = pickle.load(f)
    colorQual = pickle.load(f) 
    colorNumPix = pickle.load()
    f.close()    
else:
    import posenet, posenet_helper, pose_analyzer
    import scene_manager, camera
    import human_detector_pose_est as human_detector
    import video_channels
    import os
    trckCfg = {  # config options
        "targetPeopleDetectFPS": 2.0,
        "recordData": False, # record new data set.  if set, no algorithms will run other than capturing data at the right sampling frequency
        "runLiveGPUdetection": False,
        "num_GPU_workers": 1,
        "num_orchestrators": 2,
        "cacheDetResults": False,
        "dropFramesWhenBehind": False,  # if live capture we always may have to drop frames, but this flag, if True, will force frame dropping also when running recorded data
        "usePersistedStaticObjects": False,
        "detectStaticObjects": False,
        "doFaceID": False, 
        "runTrackerInSepProcess": False,
        # display and video options
        "doRendering": False,
        "saveVideo": True,
        "showStaticObjects": True,
        "showVisitorMap": False,
        "showGT": 2, # 0: don't show, 1: show GT boxes, 2: color objects according to correct wrt GT
        "cycleDet": False, # for example, if two cameras, ping pong between them. Since GPU is the scarce compute resource
        "globalGT": [],
        "minNumSightings": 1,
        "rec_name": [],
        "liveCapture": False # will be updated based on the channel inputs
        }

    reRunPoseAnal = True

    myChannels = [ video_channels.channel('wildtrack1','cache'), video_channels.channel('wildtrack2','cache'), video_channels.channel('wildtrack3','cache'), video_channels.channel('wildtrack4','cache'), video_channels.channel('wildtrack5','cache'),  video_channels.channel('wildtrack6','cache'),  video_channels.channel('wildtrack7','cache')]
    assert myChannels[0].hasGT, "MAKE SURE 2DGT is set to True in video_channels.py"
    import wildtrackGT
    trckCfg["globalGT"] = wildtrackGT.wildtrackGT('sampledata/wildtrack/Wildtrack_dataset/groundtruth_true_xy.txt', trckCfg["targetPeopleDetectFPS"])          

    lastFrame = 399#12 #399 

    numChannels = len(myChannels)
    trckCfg, myTracker = scene_manager.enforceConfigCoherence(trckCfg, video_channels, myChannels)
    myTracker.instrumentFNstats = True
    myTracker.doOrientation = False

    if reRunPoseAnal:
        poseSolver = pose_analyzer.poseSolver(myChannels[0].sceneMapPath)  

    detResults_all = [None] * numChannels
    imCount = 0
    savedErrors = np.zeros(0)       
    commonCats = np.zeros(0,np.bool)
    savedL1 = np.zeros(0)
    savedL1spatial = np.zeros(0)
    visibility = np.zeros(0)
    percInFrame = np.zeros(0)
    vertSizeFractions = np.zeros(0)
    Detected = np.zeros(0)
    savedLabels = np.zeros(0,np.bool) 
    vertSizeFractionsFA = np.zeros(0)
    ScoreFA = np.zeros(0)
    LabelFA = np.zeros(0,np.bool) 
    #orFeats = np.zeros((0,myTracker.numAngleBins))
    savedKL = np.zeros(0)
    colorLabel = np.zeros(0,np.bool) 
    colorQual = np.zeros(0,np.float)
    colorNumPix = np.zeros(0,np.float)

    for c in range(numChannels):
        if myChannels[c].hasGT: 
            myChannels[c].myGT.adjustGTtoFrameRate(trckCfg["targetPeopleDetectFPS"])

    numInGallery = np.max(trckCfg["globalGT"].personID) + 1
    myTracker.myTracks.posAndHeightTemp.resize(numInGallery,myTracker.myTracks.numPoseTypes+1,myTracker.myTracks.numPosDims)
    myTracker.myTracks.posCovTemp.resize(numInGallery,myTracker.myTracks.numPoseTypes+1,myTracker.myTracks.numPosDims,myTracker.myTracks.numPosDims)
    myTracker.myTracks.nConseqMisses.resize(numInGallery,numChannels)
    myTracker.myTracks.poseTypeProb.resize(numInGallery,myTracker.myTracks.numPoseTypes)
    myTracker.myTracks.poseTypeProb[:,0] = 1.0

    while imCount <= lastFrame:
        print(imCount)
        for c in range(numChannels):
            detResults_all[c] = human_detector.get_frame(imCount, os.path.dirname(myChannels[c].CapturePath)+'/cached_det_results/')
            if reRunPoseAnal:
                detResults_all[c] ["includeIt"], detResults_all[c] ["posAndHght"], detResults_all[c] ["posCov"], detResults_all[c] ["poseTypeProb"], \
                    detResults_all[c] ["boxes"], detResults_all[c] ["Orientations"], detResults_all[c] ["pose_anal_time"], detResults_all[c] ["unitVectors"] = \
                    pose_analyzer.report_detections(poseSolver, myTracker.myCamMappers[detResults_all[c] ["channelNum"]], \
                    detResults_all[c] ["keypoint_coords"], detResults_all[c] ["scores"], detResults_all[c] ["keypoint_scores"], detResults_all[c] ["num"])

        # use ground truth to populate gallery to allow creating False Negatives estimation so it too can be 
        # characterized
        
        myTracker.myTracks.Inds, dbPos = trckCfg["globalGT"].getBodyPos(imCount)
        nominalVar = np.square(0.5)
        for i2, i in enumerate(myTracker.myTracks.Inds):
            myTracker.myTracks.posAndHeightTemp[i,-1,:] = np.append(dbPos[i2,:],1.65)
            myTracker.myTracks.posCovTemp[i,0,:] = np.array([[nominalVar,0.0,0.0],[0.0,nominalVar,0.0],[0.0,0.0,np.square(0.15)]])
                
        # now we need to figure out which pose belongs to what person...
        dBmatchInd = [None] * numChannels
        goodInds = [None] * numChannels
        for c in range(numChannels):      
            goodInds[c] = np.where(detResults_all[c] ["includeIt"])[0]
            dBmatchInd[c] = myChannels[c].myGT.reportResults(imCount,detResults_all[c]["boxes"][goodInds[c],:], None)

            visib, percInFr, vertSizeFrac = myTracker.getFNprobs(c, myTracker.myTracks.Inds)
            visibility = np.append(visibility,visib)
            percInFrame = np.append(percInFrame,percInFr)
            vertSizeFractions = np.append(vertSizeFractions,vertSizeFrac)

            Detected = np.append(Detected,np.isin(myTracker.myTracks.Inds, dBmatchInd[c]))

            # now put together false positives info
            # first establish the predictors: Score & percInFr...
            vertSizeFrFA = (detResults_all[c]["boxes"][:,2] - detResults_all[c]["boxes"][:,0]) / myTracker.myCamMappers[c].n2
            Scores = detResults_all[c] ["scores"]
            
            Pos = detResults_all[c]["posAndHght"][goodInds[c],-1,0:2]
            inScope = (Pos[:,0]>-2)*(Pos[:,0]<8)*(Pos[:,1]>-5)*(Pos[:,1]<20)
            TPinds = goodInds[c][np.where((dBmatchInd[c] > -0.25)*inScope)[0]]
            FPinds = goodInds[c][np.where((dBmatchInd[c] < -0.25)*inScope)[0]]
            vertSizeFractionsFA = np.append(vertSizeFractionsFA,vertSizeFrFA[TPinds])
            ScoreFA = np.append(ScoreFA,Scores[TPinds])
            LabelFA = np.append(LabelFA,np.ones(TPinds.size))
            vertSizeFractionsFA = np.append(vertSizeFractionsFA,vertSizeFrFA[FPinds])
            ScoreFA = np.append(ScoreFA,Scores[FPinds])
            LabelFA = np.append(LabelFA,np.zeros(FPinds.size))
        for cA in range(numChannels):     
            for cB in range(cA):     
                revLookUpB = -np.ones(np.max(goodInds[cB])+1, np.int)
                revLookUpB[goodInds[cB]] = np.arange(goodInds[cB].size)
                revLookUpA = -np.ones(np.max(goodInds[cA])+1, np.int)
                revLookUpA[goodInds[cA]] = np.arange(goodInds[cA].size)
                l1, candInds = myTracker.computeLikelihoods(myTracker.z_threshold, goodInds[cA], goodInds[cB], revLookUpA, revLookUpB, detResults_all[cA]["posAndHght"], \
                    detResults_all[cA]["posCov"], detResults_all[cA]["poseTypeProb"], detResults_all[cB]["posAndHght"],detResults_all[cB]["posCov"], \
                    detResults_all[cB]["poseTypeProb"])
                for i in range(candInds.shape[0]):
                    iA = candInds[i,0]
                    iA2 = revLookUpA[iA]
                    iB = candInds[i,1]
                    iB2 = revLookUpB[iB]

                    # now look back at GT and see if this is a true match or not
                    if dBmatchInd[cA][iA2] == dBmatchInd[cB][iB2] and dBmatchInd[cA][iA2] > -0.25:
                        Match = True
                        # true match
                    elif dBmatchInd[cA][iA2] > -0.25 and dBmatchInd[cB][iB2] > -0.25:
                        # false match
                        Match = False
                    else:
                        # can't be sure so better to avoid including this one
                        continue

                    if l1[i] > 0.005:
                        combWeight = detResults_all[cA]["keypoint_scores"][iA,myTracker.poseHelper.indsForHeightEst] * detResults_all[cB]["keypoint_scores"][iB,myTracker.poseHelper.indsForHeightEst]
                        sqErr, totWeight, meanPoint = myTracker.assessPoseCompatibilities(cA,cB,detResults_all[cA]["unitVectors"][iA,:,:],detResults_all[cB]["unitVectors"][iB,:,:],combWeight, False)

                        # add features for orientation & color
                        #orFeat = myTracker.compareOrientions(detResults_all[cA]["Orientations"][iA,:],detResults_all[cB]["Orientations"][iB,:])
                        #orFeats = np.vstack((orFeats,orFeat))
                        if 0:
                            for col in range(myTracker.numChannels):
                                if detResults_all[cA]["hasColor"][iA,col] and detResults_all[cB]["hasColor"][iB,col]:
                                    # compute the KL feature that is used
                                    Cov1 = detResults_all[cB]["colorCov"][iB,col,:,:]
                                    Cov0 = detResults_all[cA]["colorCov"][iA,col,:,:]
                                    det1 = np.linalg.det(Cov1)
                                    det0 = np.linalg.det(Cov0)
                                    if det1 > 1E-16 and det0 > 1E-16: 
                                        Diff = np.array([detResults_all[cA]["colorFeats"][iA,col,:] - detResults_all[cB]["colorFeats"][iB,col,:]])
                                        invCov1 = np.linalg.inv(Cov1)
                                        # compute KL divergence
                                        KL = 0.5 * (np.trace(np.matmul(invCov1,Cov0)) + np.matmul(Diff, np.matmul(invCov1, Diff.transpose())) - myTracker.numColorFeats + np.log(det1/det0))   
                                        savedKL = np.append(savedKL,KL) 
                                        colorLabel = np.append(colorLabel,Match) 
                                        colorInd = np.append(colorInd,col) 


                        if 0:
                            invCovA = np.linalg.inv(detResults_all[cA]["posCov"][iA,-1,0:numDims,0:numDims])
                            invCovB = np.linalg.inv(detResults_all[cB]["posCov"][iB,-1,0:numDims,0:numDims])
                            Cov3 = np.linalg.inv(invCovA + invCovB)
                            jointPos = np.matmul(Cov3,np.matmul(invCovA,np.expand_dims(detResults_all[cA]["posAndHght"][iA,-1,0:numDims],axis=1))) + \
                                np.matmul(Cov3,np.matmul(invCovB,np.expand_dims(detResults_all[cB]["posAndHght"][iB,-1,0:numDims],axis=1)))
                            pointDiff = jointPos - np.expand_dims(meanPoint,axis=1)
                            zVal2 = np.matmul(pointDiff.transpose(),np.matmul(invCovA+invCovB,pointDiff))                   
                            pointDiff = np.expand_dims(detResults_all[cA]["posAndHght"][iA,-1,0:numDims] - meanPoint,axis=1)
                            zVal3 = np.matmul(pointDiff.transpose(),np.matmul(invCovA,pointDiff))    
                            pointDiff = np.expand_dims(detResults_all[cB]["posAndHght"][iB,-1,0:numDims] - meanPoint,axis=1)
                            zVal3 += np.matmul(pointDiff.transpose(),np.matmul(invCovB,pointDiff))   
                        
                        savedErrors = np.append(savedErrors,sqErr)
                        commonCats = np.append(commonCats,np.any(detResults_all[cB]["poseTypeProb"][iB,:]*detResults_all[cA]["poseTypeProb"][iA,:]))
                        savedLabels = np.append(savedLabels,Match)
                        savedL1 = np.append(savedL1,l1[i]) 
                        savedL1spatial = np.append(savedL1spatial,l1[i]) 
        
            
        if imCount > 0:
            # compare color stats with that from previous frames
            for c in range(numChannels):    
                prevValidInds = np.where((dBmatchIndPrev[c][:] > -0.25) * detResults_prev[c]["hasColor"][detResults_prev[c]["includeIt"]])[0]
                prevGoodInds = np.where(detResults_prev[c]["includeIt"])[0]
                goodInds = np.where(detResults_all[c]["includeIt"])[0]
                currentValidInds = np.where((dBmatchInd[c][:] > -0.25) * detResults_all[c]["hasColor"][detResults_all[c]["includeIt"]])[0]
                for i1 in prevValidInds:
                    I1 = prevGoodInds[i1]
                    minWght1 = np.min(detResults_prev[c] ["keypoint_scores"][I1,np.array((5,6,12,11))])
                    for i2 in currentValidInds:
                        I2 = goodInds[i2]
                        minWght2 = np.min(detResults_all[c] ["keypoint_scores"][I2,np.array((5,6,12,11))])

                        if dBmatchIndPrev[c][i1] == dBmatchInd[c][i2]:
                            Match = True
                        else: Match = False

                        # compute color divergence
                        Cov1 = detResults_all[c]["colorCov"][I2,:,:]
                        Cov0 = detResults_prev[c]["colorCov"][I1,:,:]
                        det1 = np.linalg.det(Cov1)
                        det0 = np.linalg.det(Cov0)
                        if det1 > 1E-16 and det0 > 1E-16: 
                            Diff = np.array([detResults_all[c]["colorFeats"][I2,:] - detResults_prev[c]["colorFeats"][I1,:]])
                            invCov1 = np.linalg.inv(Cov1)
                            # compute KL divergence
                            KL = 0.5 * (np.trace(np.matmul(invCov1,Cov0)) + np.matmul(Diff, np.matmul(invCov1, Diff.transpose())) - myTracker.numColorFeats + np.log(det1/det0))   
                            savedKL = np.append(savedKL,KL) 
                            colorLabel = np.append(colorLabel,Match) 
                            colorQual = np.append(colorQual,min(minWght1,minWght2)) 

                            # add number of pixels for each of these color extractions
                            numPixA = myTracker.poseHelper.getNumPixelsTorso(detResults_all[c]["keypoint_coords"][I2,:,:])
                            numPixB = myTracker.poseHelper.getNumPixelsTorso(detResults_prev[c]["keypoint_coords"][I1,:,:])
                            colorNumPix = np.append(colorNumPix,min(numPixA,numPixB)) 

        detResults_prev = detResults_all.copy()
        dBmatchIndPrev = dBmatchInd.copy()
        imCount += 1
             
    sio.savemat("intersect_error_data.mat", {'Errors':savedErrors, 'commonCats':commonCats, 'Labels':savedLabels, 'l1':savedL1, 'l1spatial':savedL1spatial, \
        'visibility':visibility, 'percInFrame':percInFrame, 'vertSizeFractions':vertSizeFractions, 'Detected':Detected, \
        'vertSizeFractionsFA':vertSizeFractionsFA,'ScoreFA':ScoreFA,'LabelFA':LabelFA, \
        'savedKL':savedKL, 'colorLabel':colorLabel, 'colorQual':colorQual, 'colorNumPix':colorNumPix  } )

    f = open("intersect_error_data.npz", "wb")
    pickle.dump(savedErrors, f)
    pickle.dump(commonCats,f)
    pickle.dump(savedLabels, f)
    pickle.dump(savedL1, f)
    pickle.dump(savedL1spatial, f)
    pickle.dump(visibility, f)
    pickle.dump(percInFrame, f)
    pickle.dump(vertSizeFractions, f)
    pickle.dump(Detected, f)
    #pickle.dump(myTracker.baseLineFN, f)
    pickle.dump(vertSizeFractionsFA, f)
    pickle.dump(ScoreFA, f)
    pickle.dump(LabelFA, f)
    #pickle.dump(orFeats, f)
    pickle.dump(savedKL, f)
    pickle.dump(colorLabel, f)
    pickle.dump(colorQual, f)
    pickle.dump(colorNumPix, f)    
    f.close()
    myTracker.shutdown() 


if analyzeIt:
    import denseNN
    optimizeFN = False
    optimizeIntersection = False
    optimizeOrientation = True

    if optimizeOrientation:
  
        #Data = np.vstack((visibility,percInFrame,vertSizeFractions)).transpose()
        denseNN.buildFNmodel1layer(orFeats,savedLabels,"orientation",num_steps=10000)
        #denseNN.buildFNmodel0layers(orFeats,savedLabels,"orientation",num_steps=10000)



    if optimizeFN:
        # first: analyze false negatives data
        
        Data = np.vstack((visibility,percInFrame,vertSizeFractions)).transpose()
        denseNN.buildFNmodel3layers(Data,Detected,"FNweights",num_steps=100000)

    if optimizeIntersection:

        lErr = np.log(savedErrors)
        lErr[np.isnan(lErr)] = 30
        Data = np.vstack((lErr,savedL1)).transpose()
        denseNN.buildFNmodel(Data,savedLabels,"intersection",num_steps=5000,numHiddenNodes = [3,3,2],regCost = 0.0)
        fff


        nBins = 50
        errBins = np.linspace(-10,4,nBins)
        goodInds = np.where((np.isnan(savedErrors)==False) * (savedErrors > -0.001))[0]
        lErr = np.log(savedErrors[goodInds])
        savedL1 = savedL1[goodInds]
        savedLabels = savedLabels[goodInds]
        probBins = np.linspace(0,1,nBins)

        inds0 = np.where(savedLabels == False)[0]
        inds1 = np.where(savedLabels == True)[0]

        # formula for likelihood ratio based on line intersections
        Left = -6.5
        Right = -2
        Top = 0.85
        Bottom = 0.03
        lErrL = (lErr < Left) * Top + (lErr > Right) * Bottom + (lErr <= Right)*(lErr >= Left)*(Right - lErr) * (Top - Bottom) / (Right - Left)
        lMod = savedL1 * lErrL / (savedL1 * lErrL + (1-savedL1) * (1-lErrL))

        PdErr = np.zeros(nBins)
        PfaErr = np.zeros(nBins)
        PdL = np.zeros(nBins)
        PfaL = np.zeros(nBins)
        PdLmod = np.zeros(nBins)
        PfaLmod = np.zeros(nBins)

        for i in range(nBins):
            PdErr[i] = 100 * np.mean(lErr[inds1] < errBins[i])
            PfaErr[i] = 100 * np.mean(lErr[inds0] < errBins[i])
            PdL[i] = 100 * np.mean(savedL1[inds1] > probBins[i])
            PfaL[i] = 100 * np.mean(savedL1[inds0] > probBins[i])   
            PdLmod[i] = 100 * np.mean(lMod[inds1] > probBins[i])
            PfaLmod[i] = 100 * np.mean(lMod[inds0] > probBins[i])  
   

        fig, axs = plt.subplots(2)
        axs[0].plot(PfaL,PdL)
        axs[0].plot(PfaErr,PdErr)
        axs[0].plot(PfaLmod,PdLmod)
    

        H1 = np.histogram(lErr[inds1],bins=errBins)[0]
        H2 = np.histogram(lErr[inds0],bins=errBins)[0]
        errMidBins = 0.5 * (errBins[0:(nBins-1)] + errBins[1:nBins]) 
        axs[1].plot(errMidBins,H1)
        axs[1].plot(errMidBins,H2)
        ax2 = axs[1].twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(errMidBins,H1/(H1 + H2),color='tab:green')
        plt.show()
import numpy as np

def solvePatch(Pos,zVar,NN,minNumPointsInPatch):
    maxNumPatches = 5
    
    numFeats = 3  # ax + by + cz = 1, a, b, c are the parameters of the plane
    scaleInPlaneVsOutOfPlane = 5
    N = zVar.size
    maxNumPatches = min(maxNumPatches,int(np.sum(NN)/minNumPointsInPatch))
    Weights = np.expand_dims(1.0 / zVar, axis = 1)
    sumWeights = np.sum(Weights)
    A = np.hstack((Pos[:,0:2],np.ones((N,1))))
    if abs(A[0,0] - 11) < 1 and abs(A[0,1]) < 5:
        d = 1
    Aw = (A * Weights).transpose()
    b = np.expand_dims(Pos[:,2],axis=1)
    Sol = [None] * maxNumPatches
    xyCenter = [None] * maxNumPatches
    Cov = [None] * maxNumPatches
    grError = [None] * maxNumPatches
    grErrorXY = [None] * maxNumPatches


    totError = np.zeros(maxNumPatches)
    BIC = np.zeros(maxNumPatches)
    oldGroup = np.zeros((N,maxNumPatches),np.int)
    smallestGroupSize = N

    for i in range(maxNumPatches):
        numGroups = i+1 
        nParam = numGroups * numFeats

        Sol[i] = [None] * numGroups
        Cov[i] = [None] * numGroups 
        Err = np.zeros((N,numGroups))
        xyErr = np.zeros((N,numGroups))
        xyCenter[i] = np.zeros((numGroups, 2))
        # just sort on z and hope for the best
        if numGroups > 1:
            sortInd = np.argsort(Pos[:,2])
            firstInd = 0
            for j in range(numGroups):
                nextLastInd = int(N * (j+1) / numGroups)
                oldGroup[sortInd[np.arange(firstInd,nextLastInd)],i] = j
                firstInd = nextLastInd                
        
        grError[i] = np.zeros(numGroups)
        grErrorXY[i] = np.zeros(numGroups)

        while True:
            for j in range(numGroups):
                # first: solve
                inds = np.where(oldGroup[:,i] == j)[0]
                Ainv = np.linalg.inv(np.matmul(Aw[:,inds],A[inds,:]))
                Sol[i][j] = np.matmul(Ainv,np.matmul(Aw[:,inds],b[inds]))        
                Cov[i][j] = Ainv
                xyCenter[i][j,:] = np.mean(Pos[inds,0:2],axis=0)
                # then compute error for the entire set
                Err[:,j] = (np.square((np.matmul(A,Sol[i][j]) - b)/1))[:,0]
                xyErr[:,j] = np.sqrt(np.sum(np.square(Pos[:,0:2] - xyCenter[i][j,:]), axis=1)) / scaleInPlaneVsOutOfPlane 
                
            # find closest cluster / group 
            newGroup = np.argmin(Err+xyErr,axis=1)
            smallestGroupSize = np.min(np.bincount(newGroup,weights=NN))
            if np.sum(np.abs(newGroup-oldGroup[:,i])) == 0 or smallestGroupSize < minNumPointsInPatch:
                break
            oldGroup[:,i] = newGroup
        totError[i] =  np.sum(Err[np.arange(N),oldGroup[:,i]] + xyErr[np.arange(N),oldGroup[:,i]]) # / sumWeights
        totError[i] =  np.sum(NN) * np.sum(Weights * Err[np.arange(N),oldGroup[:,i]] + xyErr[np.arange(N),oldGroup[:,i]])  / sumWeights


        for j in range(numGroups):
            inds = np.where(oldGroup[:,i] == j)[0]
            grError[i][j] = np.sum(Err[inds,j] * Weights[inds]) / np.sum(Weights[inds])
            grErrorXY[i][j] = np.sum(xyErr[inds,j] * Weights[inds]) / np.sum(Weights[inds])

        BIC[i] = N * np.log(totError[i] / N) + nParam*np.log(N)
        if i > 0:
            if BIC[i] > BIC[i-1] - 10:
                bestNumGrps = numGroups - 1
                break
            else:
                bestNumGrps = numGroups
        else:
            bestNumGrps = numGroups
    return Sol[bestNumGrps-1], grError[bestNumGrps-1], oldGroup[:,bestNumGrps-1]

def computePlaneError(Pos, zVar):
    Weights = np.expand_dims(1.0 / zVar, axis = 1)
    #Weights = np.ones((zVar.size,1))
    sumWeights = np.sum(Weights)
    N = Weights.size
    A = np.hstack((Pos[:,0:2],np.ones((N,1))))
    Aw = (A * Weights).transpose()
    b = np.expand_dims(Pos[:,2],axis=1)
    planeEq = np.matmul(np.linalg.inv(np.matmul(Aw,A)),np.matmul(Aw,b))
    Res = np.matmul(A,planeEq)-b
    Error = np.sum(np.square(Res)*Weights)/sumWeights
    planeCenter = np.mean(Pos[:,0:2],axis=0)
    # add a term that penalizes in-plane offset, just to encourage compact planes
    inPlaneError = 0.000001 * np.sum(np.square(Pos[:,0] - planeCenter[0]) + np.square(Pos[:,1] - planeCenter[1])) / N

    return planeEq[:,0], (Error + 0 * inPlaneError), planeCenter

def reconstruct_planar_elev_map(Pos,Cov,Scores,xLimsOuter,yLimsOuter,stepSize):

    xLimsOuter = np.array([-15.0, 20.0])
    yLimsOuter = np.array([-15.0, 35.0]) 
    stepSize = 0.1
    xBins = np.linspace(xLimsOuter[0]+stepSize/2,xLimsOuter[1]-stepSize/2,int((xLimsOuter[1]-xLimsOuter[0]) / stepSize))
    yBins = np.linspace(yLimsOuter[0]+stepSize/2,yLimsOuter[1]-stepSize/2,int((yLimsOuter[1]-yLimsOuter[0]) / stepSize))

    patchSize = 64

    inScope =  np.where((Pos[:,0]  > xLimsOuter[0]) * (Pos[:,0]  < xLimsOuter[1]) * (Pos[:,1]  > yLimsOuter[0]) * (Pos[:,1]  < yLimsOuter[1]))[0]

    xInd =  ((Pos[inScope,0] - xLimsOuter[0])/stepSize).astype(np.int)
    yInd =  ((Pos[inScope,1] - yLimsOuter[0])/stepSize).astype(np.int)

    Ind = yInd + xInd * yBins.size
    unIn,ia = np.unique(Ind, return_inverse=True)
    M = unIn.size
    meanZ = np.zeros(M)
    varZ = np.zeros(M)
    NN = np.zeros(M, np.int)
    for i2, i in enumerate(unIn):
        inds = inScope[np.where(i2 == ia)[0]]
        Weights = Scores[inds] / Cov[inds,2,2]
        sumWeights = np.sum(Weights)

        if 0 and inds.size > 2:
            sortInd = np.argsort(Pos[inds,2])
            cumSum = np.cumsum(Weights[sortInd]) / sumWeights
            # want the 50th percentile now
            try:
                Ind = np.argmax(np.where(cumSum < 0.5)[0])
                w2 = 0.5 - cumSum[Ind]
                w1 = cumSum[Ind+1] - 0.5
                meanZ[i2] = (w1 * Pos[inds[sortInd[Ind]],2] + w2 * Pos[inds[sortInd[Ind+1]],2]) / (w1 + w2)
            except:
                meanZ[i2] = Pos[inds[0],2]
        else:
            meanZ[i2] = np.sum(Pos[inds,2] * Weights) / sumWeights
            #varZ[i2] = 1.0 / sumWeights
        varZ[i2] = np.sum(np.square(Pos[inds,2] - meanZ[i2]) * Weights) / sumWeights + 1.0 / sumWeights
        NN[i2] = inds.size
        d = 1

    unxInd = unIn // yBins.size
    unyInd = unIn % yBins.size
    maxX = np.max(unxInd)
    maxY = np.max(unyInd)

    numX = int(np.ceil(xBins.size / patchSize))
    numY = int(np.ceil(yBins.size / patchSize))

    XInd = unxInd // patchSize
    YInd = unyInd // patchSize
    patchInd = YInd + XInd * numY

    unP,ia = np.unique(patchInd, return_inverse=True)
    minNumPointsInPatch = 5
    numPatches = np.zeros(0,np.int)
    smallestPatch = np.zeros(0,np.int)

    Pos2 = np.vstack((xBins[unxInd],yBins[unyInd],meanZ)).transpose()
    xGlob = np.zeros(0,np.float)
    yGlob = np.zeros(0,np.float)
    zGlob = np.zeros(0,np.float)

    patchInds = np.zeros(0,np.int)
    planeEqs = []
    patchMemberInds = []
    patchVar = np.zeros(0,np.float)
    numSplits = np.zeros(unP.size,np.int)

    for i2, i in enumerate(unP):
        inds = np.where(i2 == ia)[0]

        if np.sum(NN[inds]) >= minNumPointsInPatch:        
            planeEq, Co, grInds = solvePatch(Pos2[inds,:],varZ[inds],NN[inds], minNumPointsInPatch)

            numPatches = len(planeEq)
            numSplits[i2] = numPatches
            if numPatches > 0:
                for j in range(numPatches):
                    patchMemberInds.append(inds[np.where(grInds == j)[0]])
                    planeEqs.append(planeEq[j])
                patchInds = np.append(patchInds,i * np.ones(numPatches,np.int))                
                patchVar = np.append(patchVar, Co)


    numPatches = len(planeEqs)
    # see if some of the patches are the same plane
    absDiffs = np.zeros((numPatches,numPatches),np.float)
    meanX = np.zeros(numPatches,np.float) 
    meanY = np.zeros(numPatches,np.float) 
    patchZ = np.zeros(numPatches,np.float) 

    for i in range(numPatches):
        meanX[i] = np.mean(Pos2[patchMemberInds[i],0])
        meanY[i] = np.mean(Pos2[patchMemberInds[i],1])
        patchZ[i] = meanX[i]*planeEqs[i][0] + meanY[i]*planeEqs[i][1] + planeEqs[i][2]

    inPlaneDist = np.sqrt(np.square(np.expand_dims(meanX,axis=1) - meanX) + np.square(np.expand_dims(meanY,axis=1) - meanY))

    for i in range(numPatches):
        for j in range(numPatches):
            if i == j: 
                continue
            otherZ = meanX[i]*planeEqs[j][0] + meanY[i]*planeEqs[j][1] + planeEqs[j][2]
            absDiffs[i,j] = np.abs(patchZ[i] - otherZ) 

    absDiffs = (absDiffs + absDiffs.transpose()) / 2
    isSame = (absDiffs < 0.015) 

    numFriends = np.sum(isSame.astype(np.int),axis=1) 

    numOrigFriends = numFriends.copy()
    GroupInds = []
    if 0:
        for i in range(numPatches):
            GroupInds.append(i * np.ones(1,np.int))
        numGroups = numPatches
    else:
        Assigned = -np.ones(numPatches,np.int)
        numGroups = 0
        while np.any(Assigned == -1):
            GroupInds.append(np.zeros(0,np.int))
            seed = np.argmax(numFriends)
            GroupInds[numGroups] = np.where(isSame[seed,:])[0]
            while False:            
                Friends = np.where(np.logical_and(Assigned == -1, np.sum(isSame[GroupInds[numGroups],:],axis=0) > 0))[0]
                newFriends = Friends[np.isin(Friends,GroupInds[numGroups],invert=True)]
                if newFriends.size > 0:
                    GroupInds[numGroups] = np.append(GroupInds[numGroups],newFriends)
                else:
                    break
            Assigned[GroupInds[numGroups]] = numGroups
            numFriends[GroupInds[numGroups]] = -1
            numGroups += 1

    # next step look for groups to be merged
    PointsInGroup = [None] * numGroups
    VarInGroup = [None] * numGroups
    grPlaneEq = np.zeros((numGroups,3))
    grErr = np.zeros((numGroups,numGroups))
    BICdiff = np.zeros((numGroups,numGroups))
    #BIC  = np.zeros(numGroups)

    nInGr = np.zeros(numGroups,np.int)
    planeCenters = np.zeros((numGroups,2))
    for i in range(numGroups):  
        #nInGr[i] = GroupInds[i].size
        PointsInGroup[i] = np.zeros((0,3))
        VarInGroup[i] = np.zeros(0)
        for k in GroupInds[i]:
            x = Pos2[patchMemberInds[k],0]
            y = Pos2[patchMemberInds[k],1]
            PointsInGroup[i] = np.vstack((PointsInGroup[i], np.vstack((x, y, meanZ[patchMemberInds[k]])).transpose()))
            VarInGroup[i] = np.append(VarInGroup[i], varZ[patchMemberInds[k]])
        nInGr[i] = PointsInGroup[i].shape[0]
        Res = computePlaneError(PointsInGroup[i], VarInGroup[i])
        grPlaneEq[i,:],grErr[i,i],planeCenters[i,:] = computePlaneError(PointsInGroup[i], VarInGroup[i])



    BIC = np.zeros((numGroups,numGroups))

    for i in range(numGroups):  
        for j in range(i):
            _,grErr[i,j], _ = computePlaneError(np.vstack((PointsInGroup[i],PointsInGroup[j])), np.append(VarInGroup[i],VarInGroup[j]))
            grErr[j,i] = grErr[i,j] 
            n = nInGr[i] + nInGr[j] 
            BICdiff[i,j] = n * np.log(grErr[i,j]) + 3 * np.log(n) - (n * np.log((grErr[i,i]*nInGr[i] + grErr[j,j]*nInGr[j])/n) + 6 * np.log(n))
            BICdiff[j,i] = BICdiff[i,j]  

    oldGroup = -np.ones(M,np.int)
    while True:
        if 1:
            # do a re-assign step, kmeans style
            #first, weed out all empty groups
            goodGroups = np.where(nInGr>0)[0]
            grPlaneEq = grPlaneEq[goodGroups,:]
            numGroups = goodGroups.size
            Err = np.zeros((M,numGroups))

            for i in range(numGroups):
                Err[:,i] = np.square(grPlaneEq[i,0] * Pos2[:,0] + grPlaneEq[i,1] * Pos2[:,1] + grPlaneEq[i,2] - Pos2[:,2])
            ErrNorm = np.abs(Err) / np.expand_dims(np.sqrt(varZ),axis=1)

            possibleMember = np.exp(-ErrNorm/2)
            #possibleMember = (ErrNorm < 1.0).astype(np.int)
            Strengths = np.sum(possibleMember,axis=0)
            nCommonMembers = np.matmul(possibleMember.transpose(),possibleMember) 
            nCommonMembers[np.arange(numGroups),np.arange(numGroups)] = 0
            # make some disappear...
            # do a round of mergers
            hasBeenTouched = np.zeros(numGroups,np.bool)
            for i in range(numGroups):                
                if not hasBeenTouched[i]:
                    hasBeenTouched[i] = True
                    possibleBuddies = np.where(hasBeenTouched == False)[0]
                    if possibleBuddies.size > 0:
                        bestBuddy = possibleBuddies[np.argmax(nCommonMembers[i,possibleBuddies])] 
                        if nCommonMembers[i,bestBuddy] > 0.85 * min(Strengths[i],Strengths[bestBuddy]):
                            hasBeenTouched[bestBuddy] = True
                            if Strengths[i] < Strengths[bestBuddy]:
                                wipeOut = i
                            else:
                                wipeOut = bestBuddy
                            Err[:,wipeOut] = 1E8
            
            remainers = np.where(hasBeenTouched == False)[0]
            
            smallOnes = remainers[np.where(Strengths[remainers]/M < 0.01)[0]]
            if smallOnes.size > 0:
                Err[:,smallOnes] = 1E8 
            #subsumedOnes = np.where(np.max(nCommonMembers,axis=1) / Strengths > 0.8)[0]

            #corrCoef  = nCommonMembers / ()
            newGroup = np.argmin(Err,axis=1)
            nInGr = np.zeros(numGroups,np.int)

            for i in range(numGroups):
                inds = np.where(newGroup == i)[0]
                if inds.size > 0:
                    PointsInGroup[i] = Pos2[inds,:]
                    nInGr[i] = PointsInGroup[i].shape[0]
                    VarInGroup[i] = varZ[inds]
                    grPlaneEq[i,:],grErr[i,i], planeCenters[i,:] = computePlaneError(PointsInGroup[i], varZ[inds])
            
            if np.any(oldGroup != newGroup):
                oldGroup == newGroup
            else:
                break
            
            if 0:
                planeOffsets = 100.0 * np.ones((numGroups,numGroups))
                for i in range(numGroups):
                    for j in range(numGroups):
                        if i != j:
                            planeOffsets[i,j] = np.abs(grPlaneEq[i,0] * planeCenters[j,0] + grPlaneEq[i,1] * planeCenters[j,1] + grPlaneEq[i,2] -  \
                                (grPlaneEq[j,0] * planeCenters[j,0] + grPlaneEq[j,1] * planeCenters[j,1] + grPlaneEq[j,2]))
                planeOffsets = 0.5 * (planeOffsets + planeOffsets.transpose())

                cutoffValue = 0.25
                while True:
                    i,j = np.unravel_index(np.argmin(planeOffsets, axis=None), planeOffsets.shape)
                    if planeOffsets[i,j] > cutoffValue:
                        break
                    n = nInGr[i] + nInGr[j] 
                    _,grErr[i,j], _ = computePlaneError(np.vstack((PointsInGroup[i],PointsInGroup[j])), np.append(VarInGroup[i],VarInGroup[j]))
                    BICdiff = n * np.log(grErr[i,j]) + 3 * np.log(n) - (n * np.log((grErr[i,i]*nInGr[i] + grErr[j,j]*nInGr[j])/n) + 6 * np.log(n))
                    if BICdiff < -3.0:
                        # merge the two
                        nInGr[j] = 0
                        PointsInGroup[i] = np.vstack((PointsInGroup[i],PointsInGroup[j]))
                        nInGr[i] = PointsInGroup[i].shape[0]
                        VarInGroup[i] = np.append(VarInGroup[i], VarInGroup[j])
                        PointsInGroup[j] = np.zeros(0,np.int)
                        grPlaneEq[i,:],grErr[i,i], planeCenters[i,:] = computePlaneError(PointsInGroup[i], VarInGroup[i])

                        # update distances
                        for k in range(numGroups):
                            if k != i and nInGr[k] > 0:
                                planeOffsets[i,j] = 0.5 * (np.abs(grPlaneEq[i,0] * planeCenters[k,0] + grPlaneEq[i,1] * planeCenters[k,1] + grPlaneEq[i,2] -  \
                                        (grPlaneEq[k,0] * planeCenters[k,0] + grPlaneEq[k,1] * planeCenters[k,1] + grPlaneEq[k,2])) + \
                                        np.abs(grPlaneEq[k,0] * planeCenters[i,0] + grPlaneEq[k,1] * planeCenters[i,1] + grPlaneEq[k,2] -  \
                                        (grPlaneEq[i,0] * planeCenters[i,0] + grPlaneEq[i,1] * planeCenters[i,1] + grPlaneEq[i,2])))

                    #whether we merged or not, make sure we don't visit this combo again
                    planeOffsets[i,j] = 2.0 * cutoffValue
                    planeOffsets[j,i] = planeOffsets[i,j]

            d = 1
        else:
            # merge the most favorable one
            i,j = np.unravel_index(np.argmin(BICdiff, axis=None), BICdiff.shape)
            if BICdiff[i,j] > -0.5:
                break
            GroupInds[i] = np.append(GroupInds[i],GroupInds[j])
        
            #nInGr[i] = GroupInds[i].size
            nInGr[j] = 0
            BICdiff[:,j] = 20.0
            BICdiff[j,:] = 20.0
            PointsInGroup[i] = np.vstack((PointsInGroup[i],PointsInGroup[j]))
            nInGr[i] = PointsInGroup[i].shape[0]
            VarInGroup[i] = np.append(VarInGroup[i], VarInGroup[j])
            GroupInds[j] = np.zeros(0,np.int)
            PointsInGroup[j] = np.zeros(0,np.int)
            grPlaneEq[i,:],grErr[i,i] = computePlaneError(PointsInGroup[i], VarInGroup[i])
            for j in range(numGroups):
                if nInGr[j] > 0 and j != i:
                    _,grErr[i,j] = computePlaneError(np.vstack((PointsInGroup[i],PointsInGroup[j])), np.append(VarInGroup[i],VarInGroup[j]))
                    n = nInGr[i] + nInGr[j] 
                    BICdiff[i,j] = n * np.log(grErr[i,j]) + 3 * np.log(n) - (n * np.log((grErr[i,i]*nInGr[i] + grErr[j,j]*nInGr[j])/n) + 6 * np.log(n))
                    BICdiff[j,i] = BICdiff[i,j] 

    
    if 1:
        stepSize = 0.5
        xBins = np.linspace(xLimsOuter[0]+stepSize/2,xLimsOuter[1]-stepSize/2,int((xLimsOuter[1]-xLimsOuter[0]) / stepSize))
        yBins = np.linspace(yLimsOuter[0]+stepSize/2,yLimsOuter[1]-stepSize/2,int((yLimsOuter[1]-yLimsOuter[0]) / stepSize))


        for i in np.where(nInGr > 0)[0]:
            xInd =  ((PointsInGroup[i][:,0] - xLimsOuter[0])/stepSize).astype(np.int)
            yInd =  ((PointsInGroup[i][:,1] - yLimsOuter[0])/stepSize).astype(np.int)

            Ind = yInd + xInd * yBins.size
            unIn,ia = np.unique(Ind, return_inverse=True)

            x = ((unIn // yBins.size) + 0.5) * stepSize + xLimsOuter[0]
            y = ((unIn % yBins.size) + 0.5) * stepSize + yLimsOuter[0]


            xGlob = np.append(xGlob,x)
            yGlob = np.append(yGlob,y)
            zGlob = np.append(zGlob,grPlaneEq[i,0] * x + grPlaneEq[i,1] * y + grPlaneEq[i,2])
            
    else:

        _,_ = computePlaneError(PointsInGroup[0], VarInGroup[0])
        unP,ia = np.unique(patchInds, return_inverse=True)

        for i2, i in enumerate(unP):
            # fill in holes
            Patch = -np.ones((patchSize,patchSize),np.int)
            inds = np.where(i2 == ia)[0]
            for j in range(inds.size):
                inds2 = patchMemberInds[inds[j]]
                Patch[unyInd[inds2] - (i % numY) * patchSize, unxInd[inds2] - (i // numY) * patchSize] = j

            PatchCopy = Patch.copy()
            for k1,k2 in np.argwhere(Patch==-1):
                # look for neighbors
                ngbors = np.zeros(0,np.int)
                if k1 > 0 and Patch[k1-1,k2] > -1:
                    ngbors = np.append(ngbors, Patch[k1-1,k2])
                if k1 < patchSize - 1 and Patch[k1+1,k2] > -1:
                    ngbors = np.append(ngbors, Patch[k1+1,k2])
                if k2 > 0 and Patch[k1,k2-1] > -1:
                    ngbors = np.append(ngbors, Patch[k1,k2-1])
                if k2 < patchSize - 1 and Patch[k1,k2+1] > -1:
                    ngbors = np.append(ngbors, Patch[k1,k2+1])
                if ngbors.size > 0:
                    PatchCopy[k1,k2] = np.argmax(np.bincount(ngbors))
            Patch = PatchCopy.copy()

            # add to the global vectors now
            for j in range(inds.size):
                ks = np.argwhere(Patch == j)
                theseXinds = ks[:,1] + (i // numY) * patchSize
                theseYinds = ks[:,0] + (i % numY) * patchSize
                goodOnes = (theseXinds <= maxX) * (theseYinds <= maxY)            
                xs = xBins[theseXinds[goodOnes]]
                ys = yBins[theseYinds[goodOnes]]
                xGlob = np.append(xGlob,xs)
                yGlob = np.append(yGlob,ys)
                zGlob = np.append(zGlob,xs*planeEqs[inds[j]][0] + ys*planeEqs[inds[j]][1] + planeEqs[inds[j]][2])

    stepSize = 0.5
    im = np.zeros((yBins.size,xBins.size),np.float)
    im[((yGlob-yLimsOuter[0])/stepSize).astype(np.int),((xGlob-xLimsOuter[0])/stepSize).astype(np.int)] = zGlob
    weights = np.zeros((yBins.size,xBins.size),np.float)
    weights[((yGlob-yLimsOuter[0])/stepSize).astype(np.int),((xGlob-xLimsOuter[0])/stepSize).astype(np.int)] = 1.0

    return im.transpose().ravel(), weights.transpose().ravel(), 0.5

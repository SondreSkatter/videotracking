import numpy as np
from scipy.optimize import linear_sum_assignment
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn.cluster

def graph_separation(qualifiedMatches):

    duplicates_i = np.where(np.sum(qualifiedMatches,axis=1) > 1)[0]
    duplicates_j = np.where(np.sum(qualifiedMatches,axis=0) > 1)[0]

    # potentially, each set of duplicates is an island. But we need to check whether each i-set is connected to each j set
    n = duplicates_i.shape[0]
    m = duplicates_j.shape[0]
    subGrphsI = []
    subGrphsJ = []
    grphEdges = []
    goodMatchesA = np.zeros(0,np.int)
    goodMatchesB = np.zeros(0,np.int)
    qualifiedMatchesCopy = qualifiedMatches.copy()

    if (n > 0 or m > 0):
        opposite_is = []
        opposite_js = []
        for i in range(n):
            opposite_is.append(np.where(qualifiedMatches[duplicates_i[i],:] > 0)[0])
        for j in range(m):
            opposite_js.append(np.where(qualifiedMatches[:,duplicates_j[j]] > 0)[0])
        if (n + m > 0):
            Conn = np.zeros((n+m,n+m),np.bool)           
            for i in range(n):
                for j in range(m):
                    if (np.any(np.isin(opposite_is[i],duplicates_j[j])) or np.any(np.isin(opposite_js[j],duplicates_i[i]))):
                        Conn[i,n+j] = True
                        Conn[n+j,i] = True
            Available = np.ones(n+m,np.bool)
            nSubGraphs = 0            
            seedInd = 0
            for seedInd in range(n+m):
                if (Available[seedInd]):
                    Available[seedInd] = False
                    newSeeds = np.array([seedInd])
                    newOnes = np.array([seedInd])
                    while newSeeds.shape[0] > 0:     
                        nextSeeds = np.zeros(0,np.int)
                        for i in newSeeds:
                            toAdd = np.where(np.logical_and(Conn[i,:],Available))[0]
                            nextSeeds = np.append(nextSeeds,toAdd)
                            Available[toAdd] = False
                        newOnes = np.append(newOnes,nextSeeds)
                        newSeeds = nextSeeds
                    inds_i = np.zeros(0,np.int)
                    inds_j = np.zeros(0,np.int)
                    for i in newOnes:
                        # go back to indicies
                        if (i < n):
                            inds_i = np.append(inds_i,duplicates_i[i])
                            inds_j= np.append(inds_j,opposite_is[i])
                        else:
                            j = i - n
                            inds_i = np.append(inds_i,opposite_js[j])
                            inds_j = np.append(inds_j,duplicates_j[j])
                    inds_i = np.unique(inds_i)
                    inds_j = np.unique(inds_j)
                    grphEdges.append(qualifiedMatches[np.ix_(inds_i,inds_j)])
                    qualifiedMatches[inds_i,:] = 0
                    qualifiedMatches[:,inds_j] = 0
                    subGrphsI.append(inds_i)
                    subGrphsJ.append(inds_j)                    
                    nSubGraphs += 1
        
    for i in range(qualifiedMatches.shape[0]):
        j = np.where(qualifiedMatches[i,:])[0]
        if (j.shape[0] == 1):
            goodMatchesA = np.append(goodMatchesA, i)
            goodMatchesB = np.append(goodMatchesB, j)
    return goodMatchesA, goodMatchesB, subGrphsI, subGrphsJ, grphEdges

def solve_prob_graph(probs,minProb,qualificationRatio,minLogRatio,AScores, AisDistinct=None):
    toAdd = np.zeros(0, np.int)
    if probs.shape[1] == 0:
        leftovers = np.arange(probs.shape[0])
        allMatchesA = np.zeros(0,np.int)
        allMatchesB = np.zeros(0,np.int)
        subGrphsA = []
        subGrphsB = []
        grphEdges = []
        badSolA = []
    else:
        # First: simplify / break up the graph by identifying uncontested solutions
        qualifiedMatches = np.logical_and(probs > minProb, \
            np.logical_or(probs - qualificationRatio * np.expand_dims(np.max(probs,axis=1),axis=1) > 0.0, \
            probs - qualificationRatio * np.expand_dims(np.max(probs,axis=0),axis=0) > 0.0))
                    
        allMatchesA, allMatchesB, subGrphsA, subGrphsB, grphEdges = graph_separation(qualifiedMatches.astype(int))
        toAdd = np.zeros(0, np.int)
        zeroProb = minProb / 2 
        zeroProbPlus = zeroProb + 0.00001
        minProbLog = np.log(minProb)
        badSolA = [None] * len(subGrphsA)
        for g in range(len(subGrphsA)):
            badSolA[g] = np.zeros(0,np.int)
            if subGrphsA[g].size > 0 and subGrphsB[g].size > 0:
                NSq = max(subGrphsA[g].size, subGrphsB[g].size)
                locProb = np.zeros((subGrphsA[g].size, subGrphsB[g].size),np.float)
                locProb[:,:] = probs[np.ix_(subGrphsA[g],subGrphsB[g])] 
                            
                locProb[locProb < zeroProb] = zeroProb
                Weights = -np.log(zeroProb) * np.ones((NSq,NSq),np.float)
                Weights[0:subGrphsA[g].size,0:subGrphsB[g].size] = -np.log(locProb) 
                                    
                matchesA, matchesB = linear_sum_assignment(Weights)
                non_dummy_solutions = np.where((matchesA < subGrphsA[g].size) * (matchesB < subGrphsB[g].size))[0]
                real_A_solutions = np.where(matchesA < subGrphsA[g].size)[0]

                bestSum = np.sum(Weights[matchesA, matchesB])
                Diff = np.zeros(NSq)
                for qq in non_dummy_solutions:
                    valCopy = Weights[matchesA[qq], matchesB[qq]] 
                    Weights[matchesA[qq], matchesB[qq]] = -np.log(zeroProb)
                    resA, resB = linear_sum_assignment(Weights)
                    Diff[qq] = np.sum(Weights[resA, resB]) - bestSum
                    Weights[matchesA[qq], matchesB[qq]] = valCopy

                goodSolutions = non_dummy_solutions[np.where(Diff[non_dummy_solutions] > minLogRatio)[0]]  

                allMatchesA = np.append(allMatchesA, subGrphsA[g][matchesA[goodSolutions]])
                allMatchesB = np.append(allMatchesB, subGrphsB[g][matchesB[goodSolutions]])
                            
                nonSolutions = real_A_solutions[np.where(Diff[real_A_solutions] <= 0.0)[0]]
                if nonSolutions.size > 0:
                    if AisDistinct is None:
                        toAdd = np.append(toAdd, subGrphsA[g][nonSolutions])
                    else:
                        # we will add all badSolution if they're distinct...
                        subDistinctMat = AisDistinct[np.ix_(subGrphsA[g][nonSolutions],subGrphsA[g])]
                        isLocallyDinstinct = np.where(np.all(subDistinctMat,axis=1))[0]
                        toAdd = np.append(toAdd, subGrphsA[g][nonSolutions[isLocallyDinstinct]])

                # Deal with remaining ambiguity
                badSolutions = real_A_solutions[np.where((Diff[real_A_solutions] <= minLogRatio) * (Diff[real_A_solutions] > 0.001))[0]]

                if badSolutions.size > 0:
                    # this will be used on the outside still to identify duplicate Bs
                    badSolA[g] = matchesA[badSolutions]
                    grphEdges[g][:,matchesB[goodSolutions]] = False

        leftovers = np.where(np.any(qualifiedMatches,axis=1) == False)[0]
        if not AisDistinct is None:
            rowBetter = np.expand_dims(AScores[leftovers],axis=1) > AScores
            leftovers = leftovers[np.all(np.logical_or(AisDistinct[leftovers,:], rowBetter),axis=1)]
    toAdd = np.append(toAdd, leftovers)
    return allMatchesA, allMatchesB, toAdd, subGrphsA, subGrphsB, grphEdges, badSolA

def solve_prob_graph2(probs,probFP, probFN, qualificationRatio,minLogRatio,AScores, AisDistinct=None):
    toAdd = np.zeros(0, np.int)
    if probs.shape[1] == 0:
        leftovers = np.arange(probs.shape[0])
        allMatchesA = np.zeros(0,np.int)
        allMatchesB = np.zeros(0,np.int)
        subGrphsA = []
        subGrphsB = []
        grphEdges = []
        badSolA = []
    else:
        # First: simplify / break up the graph by identifying uncontested solutions
        qualifiedMatches = np.logical_and(probs > np.expand_dims(probFP, axis=1) * probFN, \
            np.logical_or((probs - qualificationRatio * np.expand_dims(np.max(probs,axis=1),axis=1) > 0.0), \
            probs - qualificationRatio * np.expand_dims(np.max(probs,axis=0),axis=0) > 0.0))
        allMatchesA, allMatchesB, subGrphsA, subGrphsB, grphEdges = graph_separation(qualifiedMatches.astype(int))
        
        zeroProb = np.min(np.expand_dims(probFP, axis=1) * probFN) / 2 # 2 * minProb / 2 
        zeroProbPlus = zeroProb + 0.00001
        badSolA = [None] * len(subGrphsA)
        for g in range(len(subGrphsA)):
            badSolA[g] = np.zeros(0,np.int)
            if subGrphsA[g].size > 0 and subGrphsB[g].size > 0:
                NSq = max(subGrphsA[g].size, subGrphsB[g].size)
                locProb = np.zeros((subGrphsA[g].size, subGrphsB[g].size),np.float)
                locProb[:,:] = probs[np.ix_(subGrphsA[g],subGrphsB[g])] 
                            
                locProb[locProb < zeroProb] = zeroProb
                Weights = -np.log(zeroProb) * np.ones((NSq,NSq),np.float)
                Weights[0:subGrphsA[g].size,0:subGrphsB[g].size] = -np.log(locProb) 
                                    
                matchesA, matchesB = linear_sum_assignment(Weights)
                non_dummy_solutions = np.where((matchesA < subGrphsA[g].size) * (matchesB < subGrphsB[g].size))[0]
                real_A_solutions = np.where(matchesA < subGrphsA[g].size)[0]

                bestSum = np.sum(Weights[matchesA, matchesB])
                Diff = np.zeros(NSq)
                for qq in non_dummy_solutions:
                    valCopy = Weights[matchesA[qq], matchesB[qq]] 
                    Weights[matchesA[qq], matchesB[qq]] = -np.log(zeroProb)
                    resA, resB = linear_sum_assignment(Weights)
                    Diff[qq] = np.sum(Weights[resA, resB]) - bestSum
                    Weights[matchesA[qq], matchesB[qq]] = valCopy

                goodSolutions = non_dummy_solutions[np.where(Diff[non_dummy_solutions] > minLogRatio)[0]]  

                allMatchesA = np.append(allMatchesA, subGrphsA[g][matchesA[goodSolutions]])
                allMatchesB = np.append(allMatchesB, subGrphsB[g][matchesB[goodSolutions]])
                            
                nonSolutions = real_A_solutions[np.where(Diff[real_A_solutions] <= 0.0)[0]]
                if nonSolutions.size > 0:
                    if AisDistinct is None:
                        toAdd = np.append(toAdd, subGrphsA[g][nonSolutions])
                    else:
                        # we will add all badSolution if they're distinct...
                        subDistinctMat = AisDistinct[np.ix_(subGrphsA[g][nonSolutions],subGrphsA[g])]
                        isLocallyDinstinct = np.where(np.all(subDistinctMat,axis=1))[0]
                        toAdd = np.append(toAdd, subGrphsA[g][nonSolutions[isLocallyDinstinct]])

                # Deal with remaining ambiguity
                badSolutions = real_A_solutions[np.where((Diff[real_A_solutions] <= minLogRatio) * (Diff[real_A_solutions] > 0.001))[0]]

                if badSolutions.size > 0:
                    # this will be used on the outside still to identify duplicate Bs
                    badSolA[g] = matchesA[badSolutions]
                    grphEdges[g][:,matchesB[goodSolutions]] = False

        leftovers = np.where(np.any(qualifiedMatches,axis=1) == False)[0]
        if not AisDistinct is None:
            rowBetter = np.expand_dims(AScores[leftovers],axis=1) > AScores
            leftovers = leftovers[np.all(np.logical_or(AisDistinct[leftovers,:], rowBetter),axis=1)]
    toAdd = np.append(toAdd, leftovers)
    return allMatchesA, allMatchesB, toAdd, subGrphsA, subGrphsB, grphEdges, badSolA

def hungarianDists(DistIn, maxDist):
    # Tries to minimize using Hungarian method, assuming MatrixIn is positive. Adds rows or columns if matrix is non-square
    Nq = max(DistIn.shape[0],DistIn.shape[1])
    zeroVal = 1.1 * np.max(DistIn)
    if DistIn.shape[0] == DistIn.shape[1]:
        Dist = DistIn
    else:        
        Dist = zeroVal * np.ones((Nq,Nq))
        Dist[0:DistIn.shape[0],0:DistIn.shape[1]] = DistIn
    matchesA, matchesB = linear_sum_assignment(Dist)
    non_dummy_solutions = np.where(Dist[matchesA, matchesB] < min(maxDist,zeroVal * 0.99))[0]
    return matchesA[non_dummy_solutions], matchesB[non_dummy_solutions]
    
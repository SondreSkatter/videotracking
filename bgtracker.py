import numpy as np
import time
import cv2
# Tracking the background: Detect change

def computeOverlaps(boxesA, boxesB):
    nA = boxesA.shape[0]
    nB = boxesB.shape[0]
    IoUs = np.zeros((nA,nB), np.float)
    lostAreaFraction = np.zeros((nA,nB), np.float)
    AreasA = np.multiply(boxesA[:,3] - boxesA[:,1], boxesA[:,2] - boxesA[:,0])
    AreasB = np.multiply(boxesB[:,3] - boxesB[:,1], boxesB[:,2] - boxesB[:,0])

    for i in range(nA):
        for j in range(nB):
            # check in x first since that's the skinniest dimension
            x_overlap = min(boxesA[i,3],boxesB[j,3]) - max(boxesA[i,1],boxesB[j,1])
            if (x_overlap > 0):
                x_overlap *= min(boxesA[i,2],boxesB[j,2]) - max(boxesA[i,0],boxesB[j,0])
                if (x_overlap > 0):
                    IoUs[i,j] = float(x_overlap) / float((max(boxesA[i,3],boxesB[j,3]) - min(boxesA[i,1],boxesB[j,1])) * (max(boxesA[i,2],boxesB[j,2]) - min(boxesA[i,0],boxesB[j,0]) ) )
                    lostAreaFraction[i,j] = (AreasB[j] - x_overlap) / AreasB[j]
    return IoUs, lostAreaFraction


class bgtracker:
    def __init__(self):
        d = 1
        self.numFrames = 0
        self.winSize = 20
        self.Margins0 = 20 # added space around pose bounding boxes to omit foreground
        self.MarginBottom0 = 25
        self.Boxes = np.zeros((0,4))
        self.BoxesIDs = np.zeros(0,np.int)
        self.BoxesStable = np.zeros(0,np.bool)
        self.BoxesTimeAppeared = np.zeros(0) 
        self.lastTime = 0.0
        self.lastBoxID = -1

    def addFrame(self, frame, boxes, imScaleIn, timeNow):
        timeStart = time.time()
        Scale = 0.35
        self.Margins = int(self.Margins0 * Scale)
        self.MarginBottom = int(self.MarginBottom0 * Scale)
        self.imScale = imScaleIn / Scale
        im = cv2.resize(frame, (int(frame.shape[1] * Scale),int(frame.shape[0] * Scale)), interpolation = cv2.INTER_LINEAR).astype(np.float)/255

        minNumPix = int(5E-4 * im.size) #   300
        if self.numFrames == 0:
            # Now we know the size            
            self.pixMeans = np.zeros(im.shape)
            self.pixMeansAlt = np.zeros(im.shape)
            #self.pixCov = np.zeros((frame.shape[0],frame.shape[1],3,3))
            self.numObs = np.zeros((im.shape[0],im.shape[1]),np.int)
            self.numObsAlt = np.zeros((im.shape[0],im.shape[1]),np.int)
            self.backGround = np.ones((im.shape[0],im.shape[1]),np.bool)
            self.numInBg = np.zeros((im.shape[0],im.shape[1]),np.int)

        self.backGround = np.ones((im.shape[0],im.shape[1]),np.bool)

        # except the areas covered by people
        for i in range(boxes.shape[0]):
            self.backGround[max(0,int(boxes[i,0]/self.imScale[0])-self.Margins):min(im.shape[0]-1,int(boxes[i,2]/self.imScale[0])+self.MarginBottom),
                max(0,int(boxes[i,1]/self.imScale[1])-self.Margins):min(im.shape[1]-1,int(boxes[i,3]/self.imScale[1])+self.Margins)] = False
            
        self.numInBg += self.backGround 
        self.Thres = 0.2
        Diff2 = np.sum(np.square(im  - self.pixMeans), axis = 2)
        

        if 0 and self.numFrames > 0:
            import matplotlib.pyplot as plt
            plt.hist(Diff.ravel(), bins='auto') 
            plt.show()
        
        #toUpdate = self.backGround * np.logical_or(self.numObs == 0, Diff2 < self.Thres**2)
        toUpdate = np.logical_or(self.backGround * (self.numObs == 0), (self.numObs > 0) * (Diff2 < self.Thres**2))

        oldFg = np.argwhere(self.numObsAlt > 0)
        fgDiff2 = np.sum(np.square(im[oldFg[:,0],oldFg[:,1],:]  - self.pixMeansAlt[oldFg[:,0],oldFg[:,1],:]), axis = 1)
        oldFg = oldFg[fgDiff2 < Diff2[oldFg[:,0],oldFg[:,1]],:]
        toUpdate[oldFg[:,0],oldFg[:,1]] = False

        
        newMix = 1.0 / (1.0 + np.minimum(self.numObs[toUpdate],self.winSize)) 
        self.pixMeans[toUpdate] = self.pixMeans[toUpdate] * np.expand_dims(1.0-newMix,axis=1) + np.expand_dims(newMix,axis=1) * im[toUpdate]
        self.numObs[toUpdate] += 1
        self.numObsAlt[toUpdate] = 0  # This one is probably a little too harsh... one mistake, and it'
        # reset

        toUpdateAlt = np.argwhere(self.backGround * (toUpdate == False))
        altDiff = np.linalg.norm(im[toUpdateAlt[:,0],toUpdateAlt[:,1],:]  - self.pixMeansAlt[toUpdateAlt[:,0],toUpdateAlt[:,1],:], axis = 1)

        possibleResets = toUpdateAlt[altDiff >= self.Thres,:]
        if possibleResets.size > 0:
            self.numObsAlt[possibleResets[:,0],possibleResets[:,1]] = 0
            if 0:
                resetBaseline = possibleResets[np.where(self.numObs[possibleResets[:,0],possibleResets[:,1]] < self.winSize)[0],:]
                if resetBaseline.size > 0:                
                    self.pixMeans[resetBaseline[:,0],resetBaseline[:,1],:] = im[resetBaseline[:,0],resetBaseline[:,1],:]
                    self.numObs[resetBaseline[:,0],resetBaseline[:,1]] = 1

        newMix = 1.0 / (1.0 + np.minimum(self.numObsAlt[toUpdateAlt[:,0],toUpdateAlt[:,1]],self.winSize)) 
        self.pixMeansAlt[toUpdateAlt[:,0],toUpdateAlt[:,1],:] *= np.expand_dims(1.0 - newMix,axis=1)
        self.pixMeansAlt[toUpdateAlt[:,0],toUpdateAlt[:,1],:] += np.expand_dims(newMix,axis=1) * im[toUpdateAlt[:,0],toUpdateAlt[:,1],:]
        self.numObsAlt[toUpdateAlt[:,0],toUpdateAlt[:,1]] += 1

        Swappers = np.where((self.numObsAlt[toUpdateAlt[:,0],toUpdateAlt[:,1]] > self.numObs[toUpdateAlt[:,0],toUpdateAlt[:,1]]) *
            (self.numObs[toUpdateAlt[:,0],toUpdateAlt[:,1]] < 5))[0]        
        if Swappers.size > 0:
            self.pixMeans[toUpdateAlt[Swappers,0],toUpdateAlt[Swappers,1],:] = self.pixMeansAlt[toUpdateAlt[Swappers,0],toUpdateAlt[Swappers,1],:] 
            self.numObs[toUpdateAlt[Swappers,0],toUpdateAlt[Swappers,1]] = self.numObsAlt[toUpdateAlt[Swappers,0],toUpdateAlt[Swappers,1]] 
            self.numObsAlt[toUpdateAlt[Swappers,0],toUpdateAlt[Swappers,1]] = 0

        if self.numFrames > self.winSize and self.numFrames % 5 == 0:
            # some maintenane in the end... cleaning out backgrounds around areas where a person has blocked things most of the time...
            FalseBg = (self.backGround == False) * (self.numInBg / self.numFrames < 0.25)    
            self.numObs[FalseBg] = 0
            self.numObsAlt[FalseBg] = 0

        # Lastly, detect objects...
        fg = (self.numObsAlt > 1).astype(np.uint8)
        kernel = np.ones((3,3),np.uint8)
        #self.opening = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        self.opening = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel), cv2.MORPH_DILATE,kernel)


        newBoxes = np.zeros((0,4),np.int)
        
        labelMask = None
        if np.sum(self.opening) > minNumPix:
            connectivity = 4
            numObj,labelMask, BBs, Centroids = cv2.connectedComponentsWithStats(self.opening, connectivity, cv2.CV_32S)
            objSizesNumPx = BBs[1:,4]
            BagObj = np.where(objSizesNumPx > minNumPix)[0]
            
            if BagObj.size > 0:
                newBoxes = np.zeros((BagObj.size,4))
                for i2, i in enumerate(BagObj):
                    newBoxes[i2,:] = np.array((BBs[i+1,1]*self.imScale[0],BBs[i+1,0]*self.imScale[1],
                        self.imScale[0]*(BBs[i+1,1]+BBs[i+1,3]),self.imScale[1]*(BBs[i+1,0]+BBs[i+1,2])))
        
        # now compare new boxes, if any with the old
        if newBoxes.shape[0] > 0 or self.BoxesIDs.size > 0:
            if newBoxes.shape[0] == 0:
                # boxes disappeared, just remove them then
                self.Boxes.resize((0,4))
                self.BoxesIDs.resize(0)
                self.BoxesTimeAppeared.resize(0) 
                self.BoxesStable.resize(0) 
                self.BoxesLastLabel.resize(0) 
                self.BoxesNumPix.resize(0)  
            elif self.BoxesIDs.size == 0:
                # new boxes, but no old ones to match them to
                self.BoxesIDs = np.arange(self.lastBoxID+1,self.lastBoxID+1+BagObj.size)
                self.BoxesLastLabel = BagObj.copy()
                self.BoxesNumPix = objSizesNumPx[BagObj]
                self.lastBoxID += BagObj.size
                self.BoxesTimeAppeared = np.ones(BagObj.size) * 0.5 * (self.lastTime + timeNow)
                self.Boxes = newBoxes.copy()
                self.BoxesStable.resize(BagObj.size) # they will have Stable=False to begin with...
            else:
                # reconcile old and new boxes
                Overlaps, lostAreaFraction = computeOverlaps(newBoxes,self.Boxes)
                matchInds = np.argmax(Overlaps,axis=1)
                iouThres = 0.15
                iouStableThres = 0.9
                goodMatches = Overlaps[np.arange(BagObj.size), matchInds] > iouThres

                # Test these matches a little more thoroughly

                stableMatch = np.zeros(goodMatches.size)
                for i in np.where(goodMatches)[0]:
                    #i = goodMatches[i2]
                    j = matchInds[i2]
                    minY = int(min(newBoxes[i,0],self.Boxes[j,0]) / self.imScale[0])
                    maxY = int(max(newBoxes[i,2],self.Boxes[j,2])/ self.imScale[0])
                    minX = int(min(newBoxes[i,1],self.Boxes[j,1]) / self.imScale[1])
                    maxX = int(max(newBoxes[i,3],self.Boxes[j,3]) / self.imScale[0])
                    overlap = np.sum(((labelMask[minY:maxY,minX:maxX] == BagObj[i]+1) * (self.lastLabel[minY:maxY,minX:maxX] == self.BoxesLastLabel[j]+1) ).astype(np.int))
                    areaLoss = (self.BoxesNumPix[j] - overlap) / self.BoxesNumPix[j]
                    stableMatch[i] = areaLoss < 0.1


                # 1) update the coordinates for the ones that match
                self.Boxes = newBoxes[goodMatches,:]
                self.BoxesLastLabel = BagObj[goodMatches]
                self.BoxesNumPix = objSizesNumPx[BagObj[goodMatches]]
                # 2) remove the old boxes where there is currently no match
                self.BoxesTimeAppeared = self.BoxesTimeAppeared[matchInds[goodMatches]]
                self.BoxesIDs = self.BoxesIDs[matchInds[goodMatches]]
                #self.BoxesStable = self.BoxesStable[matchInds[goodMatches]]
                self.BoxesStable = (Overlaps[np.where(goodMatches)[0], matchInds[goodMatches]] > iouStableThres) * stableMatch[goodMatches]

                # add new ones
                newOnes = np.where(goodMatches == False)[0]
                if newOnes.size > 0:
                    self.Boxes = np.vstack((self.Boxes,newBoxes[newOnes,:]))
                    self.BoxesIDs = np.append(self.BoxesIDs, np.arange(self.lastBoxID+1,self.lastBoxID+1+newOnes.size))
                    self.lastBoxID += newOnes.size
                    self.BoxesTimeAppeared = np.append(self.BoxesTimeAppeared, np.ones(newOnes.size) * 0.5 * (self.lastTime + timeNow))
                    self.BoxesStable = np.append(self.BoxesStable, np.zeros(newOnes.size, np.bool))
                    self.BoxesLastLabel = np.append(self.BoxesLastLabel, BagObj[newOnes])
                    self.BoxesNumPix = np.append(self.BoxesNumPix, objSizesNumPx[BagObj[newOnes]])

        self.numFrames += 1
        self.lastTime = timeNow
        self.lastLabel = labelMask
        print('Time to process background: {:.2f} msec'.format(1000*(time.time()-timeStart)))
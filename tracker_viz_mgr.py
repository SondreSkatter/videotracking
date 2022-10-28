import cv2
import numpy as np
import time
import posenet_helper
import re
import matplotlib.pyplot as plt
import textwrap 
import platform

class tracker_viz_mgr:
    def __init__(self, Channels, trckCfg, myCamMappers = None, combinedDisplay = True, interactivePlot = False):
        self.numChannels = len(Channels)
        self.keepAspectRatios = True
        self.myChannels = Channels
        self.interactivePlot = interactivePlot
        self.showGT = trckCfg["showGT"] # 0: don't show, 1: show GT boxes, 2: color objects according to correct wrt GT
        self.combinedDisplay = combinedDisplay
        self.saveVideo = trckCfg["saveVideo"]
        self.globalGT = trckCfg["globalGT"] 
        self.showVisitorMap = False # will just switch on if there is data
        self.myCamMappers = myCamMappers

        if interactivePlot:
            self.posenet_helper = posenet_helper.posenet_helper()
            self.objRef = [None] * self.numChannels

        # determine the layout of combined window
        if combinedDisplay:
            # Need to deal with potentially different aspect ratios       
            channelDims = np.zeros((self.numChannels,2),np.int)
            for i in range(self.numChannels):
                channelDims[i,:] = self.myChannels[i].imSize[0:2]
            self.avAspectRatio = np.sum(channelDims[:,1]) / np.sum(channelDims[:,0]) 
            self.aspectRatios = channelDims[:,1] / channelDims[:,0]
            # if the images have different aspect ratio we will use the average one to allocate screen space
            # the gaps can then be either left empty, or image scaled to wrong aspect ratio
            # Aiming for a 1920:1080 ratio of the total image now
            combTargetRatio = 1920/1080
            combinedRatio = np.zeros(self.numChannels)
            for j in range(self.numChannels):
                # evaluate different row & column paritionings
                nX = j + 1
                nY = int(np.ceil(self.numChannels / nX))   
                combinedRatio[j] = nX * self.avAspectRatio / nY
            bestChoice = np.argmin(np.abs(np.log(combinedRatio)-np.log(self.avAspectRatio)))
            self.nX = bestChoice + 1
            self.nY = int(np.ceil(self.numChannels / self.nX))      
            AspectRatio = self.avAspectRatio * self.nX / self.nY
        else:
            self.fontSize = 0.7 * Channels[0].imSize[0] / 1080 
            self.channelSwitchFreq = 5  

        cv2.namedWindow('cameraview', cv2.WINDOW_NORMAL)
        if self.saveVideo:
            targetVideoSize = np.array((1080,1920),np.int) # if ratio different, then have the larger dimension equal the target value
            self.videoHeightPixels = targetVideoSize[0]
            if platform.system() == 'Windows':
                fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
            else:
                fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            if combinedDisplay:
                if AspectRatio * targetVideoSize[0] > targetVideoSize[1]:
                    self.videoHeightPixels = targetVideoSize[0]
                    self.videoWidthPixels = int(AspectRatio*self.videoHeightPixels)
                else:
                    self.videoWidthPixels = targetVideoSize[1]
                    self.videoHeightPixels = int(self.videoWidthPixels / AspectRatio)
            else:
                self.videoWidthPixels = int(Channels[0].imSize[1]*self.videoHeightPixels/Channels[0].imSize[0])
            self.out = cv2.VideoWriter('output.mp4', fourcc, trckCfg["targetPeopleDetectFPS"], (self.videoWidthPixels, self.videoHeightPixels))

        self.Colors = np.array([[255,0,0],[0,0,255],[0,255,0],[255,128,0],[153,51,255],[255,255,0],[255,0,127],[51,255,153]],int)
        self.imBufferInitialized = False

    def close(self):
        cv2.destroyWindow('cameraview')
        if self.saveVideo:
            self.out.release()
        if self.showVisitorMap:
            cv2.destroyWindow('visitorview')

    def initBuffers(self, imgBuff):
        # this should be called upon receiving the first images; since the images may have been downscaled
        # we don't necessarily know the size at init time
        if self.combinedDisplay:
            channelDims = np.zeros((self.numChannels,2),np.int)
            for i in range(self.numChannels):
                channelDims[i,:] = imgBuff[i].shape[0:2]

            if self.keepAspectRatios:                
                self.avAspectRatio = np.sum(channelDims[:,1]) / np.sum(channelDims[:,0])  # might have changed slightly after the rescale due tot the stride issue in pose est
                self.aspectRatios = channelDims[:,1] / channelDims[:,0]
            else:
                self.cmnImSz = np.mean(channelDims,axis=0).astype(np.int)        
            
            # determine the max width for each column
            leftPosAccum = 0
            self.colPos = np.zeros((self.nX,2),np.int)
            Inds = np.arange(self.numChannels)
            for i in range(self.nX):
                self.colPos[i,0] = leftPosAccum
                if self.keepAspectRatios:
                    leftPosAccum += np.max(channelDims[np.where(Inds % self.nX == i)[0],1])
                else:
                    leftPosAccum += self.cmnImSz[1]
                self.colPos[i,1] = leftPosAccum - 1
            topPosAccum = 0
            self.rowPos = np.zeros((self.nY,2),np.int)
            Inds = np.arange(self.numChannels)
            for i in range(self.nY):
                self.rowPos[i,0] = topPosAccum
                if self.keepAspectRatios:
                    matches = np.where(Inds // self.nX == i)[0]
                    if matches.size > 0:
                        topPosAccum += np.max(channelDims[matches,0])
                    else:
                        topPosAccum += self.rowPos[i-1,1] - self.rowPos[i-1,0] + 1
                else:
                    topPosAccum += self.cmnImSz[0]
                self.rowPos[i,1] = topPosAccum - 1

            if self.keepAspectRatios:
                self.imNeedsRescaling = np.zeros(self.numChannels,np.bool)
            else:
                self.imNeedsRescaling = np.ones(self.numChannels,np.bool)
                for c in range(self.numChannels):
                    if imgBuff[c].shape[0] == self.cmnImSz[0] and imgBuff[c].shape[1] == self.cmnImSz[1]:
                        self.imNeedsRescaling[c] = False

            # determine the size of combined window
            self.imDisplay = np.zeros((topPosAccum,leftPosAccum,3),np.uint8)
            self.fontSize = self.imDisplay.shape[0] / 1080 
            if self.interactivePlot:
                self.indexImAll = np.zeros(self.imDisplay.shape, np.uint8)
        
        if self.interactivePlot:
            self.indexIm = [None] * self.numChannels
            for c in range(self.numChannels):
                self.indexIm[c] = np.zeros(imgBuff[c].shape, np.uint8)                
        self.imBufferInitialized = True
        self.drawThickness = int(imgBuff[0].shape[0] / 1080 * 2)


    def renderFrame(self,fr, trackingResults, detResults, imgBuff,correct_det2D,keepGoing,
            correct_det3D=None,haltFrame=False,GTitemsMissed=None):

        if not self.imBufferInitialized:
            self.initBuffers(imgBuff)

        # Show pose overlays of the tracked humans        
        goodMatches = trackingResults["goodMatches"]
        Description = trackingResults["Description"]
        freshChannels = trackingResults["freshChannels"]
        self.channelToDisplay = 0
        errorInFrame = False
        if not self.combinedDisplay:
            self.channelToDisplay = (fr // self.channelSwitchFreq) % self.numChannels 

        hasLeftLuggageInfo = "leftLuggageBBs" in trackingResults
        if hasLeftLuggageInfo:
            bagLeavers = np.unique(trackingResults["leftLuggageOwnerID"][:,0])

        for c in range(self.numChannels):
            Ind = c 
            Res = detResults[c]

            if hasLeftLuggageInfo:
                for ii in range(trackingResults["leftLuggageInds"][c].size):
                    # draw that box
                    cv2.rectangle(imgBuff[c],(
                        int((trackingResults["leftLuggageBBs"][c][ii,1]-Res["clipOffset"][1])/Res["output_scale"][1]),
                        int((trackingResults["leftLuggageBBs"][c][ii,0]-Res["clipOffset"][0])/Res["output_scale"][0])),
                        (int((trackingResults["leftLuggageBBs"][c][ii,3]-Res["clipOffset"][1])/Res["output_scale"][1]),
                        int((trackingResults["leftLuggageBBs"][c][ii,2]-Res["clipOffset"][0])/Res["output_scale"][0])),
                        (0,0,255),2)    

            if np.any(freshChannels == c):
                scale4 = np.array((Res["output_scale"][0],Res["output_scale"][1],Res["output_scale"][0],Res["output_scale"][1]))
                if not 'clipOffset' in Res.keys():
                    # Just for backward compatibility with old movies
                    Res["clipOffset"] = np.zeros(2,np.int)
                offset4 = np.append(Res["clipOffset"],Res["clipOffset"])
                thisDrawThickness = self.drawThickness 
                thisFontSize  = self.fontSize / np.mean(Res["output_scale"])
                if self.interactivePlot:
                    self.indexIm[c].fill(0)
                    #self.trackingResults = trackingResults
                    self.detResults = detResults
                    self.objRef[c] = np.zeros(Res["num"],np.int)
                for i in range(len(Description[c])):          
                    i3 = int(goodMatches[c][i])
                    theseKeypoints = (Res["keypoint_coords"][i3,:,:] - Res["clipOffset"]) / Res["output_scale"]
                    theseKeypointScores = Res["keypoint_scores"][i3,:]
                    
                    box = ((Res["boxes"][i3,:].astype(np.float) - offset4)  / scale4).astype(np.int)

                    matchObject = re.match('(\d{1,})([a-z])',Description[c][i]).group(1)                    

                    colorInd = int(matchObject) % self.Colors.shape[0]
                    color = (int(self.Colors[colorInd,2]),int(self.Colors[colorInd,1]),int(self.Colors[colorInd,0]))

                    if hasLeftLuggageInfo:
                        if bagLeavers.size > 0:
                            if np.isin(int(matchObject),bagLeavers):    
                                AddedText = textwrap.wrap("Alert! This person just left that bag!", width=20)
                                textsize = cv2.getTextSize(AddedText[0], cv2.FONT_ITALIC,  2 * thisFontSize, thisDrawThickness*2)[0]
                                gap = textsize[1] + 10
                                txtWdth = textsize[0]  
                                numLines = len(AddedText)
                                posX = int(box[1] - txtWdth/2)
                                posY = max(gap,int(box[0] - (numLines - 0.5) * gap))

                                if posX + txtWdth/2 > imgBuff[c].shape[1]:
                                    posX -= int(txtWdth / 2) 
                                elif posX < txtWdth/2:
                                    posX += int(txtWdth / 2) 
                                for line in AddedText:
                                    cv2.putText(imgBuff[c],line,(posX,posY), cv2.FONT_ITALIC, 2 * thisFontSize,color, thisDrawThickness*2)
                                    posY += gap
                    
                    showAnnotationAndSkeleton = 1
                    if showAnnotationAndSkeleton:
                        cv2.putText(imgBuff[c],Description[c][i],(int(box[1]),int(box[0])), cv2.FONT_ITALIC, thisFontSize, color, thisDrawThickness*2)
                        posenet_helper.draw_single_skel_and_kp(imgBuff[c], Res["scores"][i3], theseKeypointScores, theseKeypoints, color, min_pose_score=0.12, min_part_score=0.1)
                    if self.interactivePlot:
                        self.objRef[c][i3] = matchObject
                        posenet_helper.draw_single_skel_and_kp(self.indexIm[c], Res["scores"][i3], theseKeypointScores, theseKeypoints, (i3+1,i3+1,i3+1), min_pose_score=0.12, min_part_score=0.1, forIndexImage = True)

                    if self.showGT == 2 and self.myChannels[c].hasGT and not self.globalGT: 
                        if correct_det2D[c][i]:
                            GTcolor = (0,255,0)
                        else:
                            GTcolor = (0,0,255)

                        cv2.rectangle(imgBuff[c],(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),GTcolor,2)

                    if "staticObjects" in trackingResults.keys():                         
                        static_boxes = trackingResults["staticObjects"][c]
                        for i in range(static_boxes.shape[0]):
                            box = (static_boxes[i,:] - offset4)/ scale4
                            cv2.rectangle(imgBuff[c],(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(255,0,0), thisDrawThickness)
            
                    if self.showGT == 1 and self.myChannels[c].hasGT : 
                        dBIDs,dbBBs = self.myChannels[c].myGT.getBodyBB(fr)
                        for i in range(dbBBs.shape[0]):
                            box = (dbBBs[i,:] - offset4)/ scale4
                            cv2.rectangle(imgBuff[c],(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(0,255,0), thisDrawThickness)

                    if self.globalGT and self.showGT > 0:
                        # use correct_det: # 0: not scored, 1: correct det, -2: false positive, -1: mistracked
                        matchedGlobal = np.where(int(matchObject) == trackingResults["Inds"])[0]
                        if matchedGlobal.size == 1:
                            correctness = correct_det3D[0][matchedGlobal]
                            if correctness != 0:
                                if correctness == 1:
                                    # draw green circle
                                    CircCol = (0,255,0)
                                else:
                                    # draw red circle
                                    CircCol = (0,0,255)
                                    if correctness == -1:
                                        Txt = "FT:"
                                    else:
                                        Txt = "FP"
                                    Txt += " " + correct_det3D[2][matchedGlobal[0]]
                                    # draw text
                                    cv2.putText(imgBuff[c],Txt,(int(box[1]),int(box[0])+20), cv2.FONT_ITALIC, thisFontSize, CircCol, thisDrawThickness*2)

                                cv2.circle(imgBuff[c],(int(box[1]),int(box[0])),10,CircCol,thickness=-1)

                if self.globalGT and self.showGT > 0: # and self.myChannels[c].hasGT:
                    # draw the misses also...
                    # 1) find all misses
                    dBmisses = correct_det3D[1]

                    if 1:
                        for i2, i in enumerate(dBmisses):
                            Point = np.array((correct_det3D[4][i2,0],correct_det3D[4][i2,1],0.0))
                            if correct_det3D[4].shape[1] == 3:
                                Height = correct_det3D[4][i2,2]
                            else:
                                Height = 1.65
                            transVec = np.cross(self.myCamMappers[c].trans - Point,np.array((0,0,1)))
                            transVec /= np.linalg.norm(transVec)
                            halfWidth = 0.45 / 2

                            Points = np.vstack((Point - transVec*halfWidth,Point + transVec*halfWidth,
                                Point + transVec*halfWidth + np.array((0.0,0.0,Height)),Point - transVec*halfWidth + np.array((0.0,0.0,Height)))).transpose()
                            cornersX,cornersY = self.myCamMappers[c].get_pixel_from_world_pos_new(Points)
                            Points = np.expand_dims(np.vstack(((cornersX-offset4[1])/scale4[1],(cornersY-offset4[0])/scale4[0])).astype(np.int32).transpose(),axis=1)
                            cv2.polylines(imgBuff[c],[Points],True,(0,0,255), self.drawThickness * 2) 
                            cv2.putText(imgBuff[c],correct_det3D[3][i2],(int(np.min(Points[:,0,0])),int(np.min(Points[:,0,1]))), cv2.FONT_ITALIC, thisFontSize, (0,0,255), thisDrawThickness*2)
                    else:

                        # 2) find these misses in the 2DGT (if it exist
                        dBIDs,dbBBs = self.myChannels[c].myGT.getBodyBB(fr)
                        theseMisses = np.isin(dBIDs,dBmisses)
                        for i2, i in enumerate(theseMisses):
                            box = dbBBs[i,:] / scale4
                            cv2.rectangle(imgBuff[c],(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),(0,0,255), thisDrawThickness*2)
                            cv2.putText(imgBuff[c],correct_det3D[3][i2],(int(box[1]),int(box[0])+20), cv2.FONT_ITALIC, thisFontSize, CircCol, thisDrawThickness*2)

            if imgBuff[c] != []:
                if (self.combinedDisplay and c == 0) or (not self.combinedDisplay and c == self.channelToDisplay):
                    # also, write the frame number
                    cv2.putText(imgBuff[c],str(fr),(int(50*imgBuff[c].shape[0]/1080),int((1080-50)*imgBuff[c].shape[0]/1080)), cv2.FONT_ITALIC, self.fontSize, (0,255,255), int(self.drawThickness * 2))
                if self.combinedDisplay:
                    thisCol = c%self.nX
                    thisRow = c//self.nX

                    if self.imNeedsRescaling[c]:
                        self.imDisplay[np.ix_(np.arange(self.rowPos[thisRow,0],self.rowPos[thisRow,1]),np.arange(self.colPos[thisCol,0],self.colPos[thisCol,1]),np.arange(3))] = \
                            cv2.resize(imgBuff[c],(0,0),fx=(self.colPos[thisCol,1]-self.colPos[thisCol,0])/imgBuff[c].shape[1],fy=(self.rowPos[thisRow,1]-self.rowPos[thisRow,0])/imgBuff[c].shape[0],interpolation = cv2.INTER_AREA)
                    else:
                        self.imDisplay[np.ix_(np.arange(self.rowPos[thisRow,0],self.rowPos[thisRow,0]+imgBuff[c].shape[0]),np.arange(self.colPos[thisCol,0],self.colPos[thisCol,0]+imgBuff[c].shape[1]),np.arange(3))] = imgBuff[c]

                    if self.interactivePlot:
                        if self.imNeedsRescaling[c]:
                            self.indexImAll[np.ix_(np.arange(self.rowPos[thisRow,0],self.rowPos[thisRow,1]),np.arange(self.colPos[thisCol,0],self.colPos[thisCol,1]))] = \
                                cv2.resize(self.indexIm[c],(0,0),fx=(self.colPos[thisCol,1]-self.colPos[thisCol,0])/imgBuff[c].shape[1],fy=(self.rowPos[thisRow,1]-self.rowPos[thisRow,0])/imgBuff[c].shape[0],interpolation = cv2.INTER_NEAREST)
                        else:
                            self.indexImAll[np.ix_(np.arange(self.rowPos[thisRow,0],self.rowPos[thisRow,0]+imgBuff[c].shape[0]),np.arange(self.colPos[thisCol,0],self.colPos[thisCol,0]+imgBuff[c].shape[1]))] = self.indexIm[c]

        if not self.combinedDisplay:
            self.imDisplay = imgBuff[self.channelToDisplay]
        cv2.imshow('cameraview', self.imDisplay)
        key = cv2.waitKey(1)
        if self.saveVideo:
            self.out.write(cv2.resize(self.imDisplay, (0,0), fy=self.videoHeightPixels/self.imDisplay.shape[0], fx=self.videoWidthPixels/self.imDisplay.shape[1],interpolation = cv2.INTER_AREA))
                    
        if "visitorMap" in trackingResults.keys(): 

            if not self.showVisitorMap:
                cv2.namedWindow('visitorview', cv2.WINDOW_NORMAL)
            self.showVisitorMap = True
            visitorMap = trackingResults["visitorMap"][self.channelToDisplay] 
            cv2.imshow('visitorview', visitorMap / np.max(visitorMap))
            key = cv2.waitKey(1)
        if self.interactivePlot:
            cv2.setMouseCallback("cameraview", self.click)
        if haltFrame:
            # wait until 'n' is pressed before moving on to the next frame
            while True:
                key = cv2.waitKey(1)
                if key & 0xFF == ord('n'): 
                    return 1
                elif key & 0xFF == ord('q'):
                    return 0
                elif key & 0xFF == ord('p'):
                    return -1
                time.sleep(0.05)
        else:
            if key & 0xFF == ord('q'): keepGoing = False 
            return keepGoing

    def click(self, event, x, y, flags, param):
        try:
            if self.combinedDisplay:
                col = np.max(np.where(x>self.colPos[:,0])[0])
                row = np.max(np.where(y>self.rowPos[:,0])[0])
                ch = col + self.nX * row
                objNum = self.indexImAll[y,x,0]
            else:    
                ch = self.channelToDisplay
                objNum = self.indexIm[ch][y,x,0]
        except:
            print("hmm, x y position outside of image. ")
            return

        if objNum > 0:
            objNum -= 1
            globObj = self.objRef[ch][objNum]
            img = self.imDisplay.copy()
            Text = 'Obj num: {:d}'.format(globObj)
            colorInd = globObj % self.Colors.shape[0]
            color = (int(self.Colors[colorInd,2]),int(self.Colors[colorInd,1]),int(self.Colors[colorInd,0]))
           
            if event == cv2.EVENT_LBUTTONDOWN:
                # show some stats too            
                self.frameNum = self.detResults[ch]["frameNum"]
                numColPix = self.posenet_helper.getNumPixelsTorso(self.detResults[ch]["keypoint_coords"][objNum,:,:])
                self.showStats(globObj,objNum,self.detResults[ch]["posAndHght"][objNum,:,:], self.detResults[ch]["posCov"][objNum,:,:,:], self.detResults[ch]["poseTypeProb"][objNum,:], \
                   self.detResults[ch]["scores"][objNum], self.detResults[ch]["Orientations"][objNum,:], numColPix)  
            cv2.putText(img,Text,(x,y), cv2.FONT_ITALIC,  1,color,4)
        else:
            img = self.imDisplay
        cv2.imshow('cameraview', img)
        key = cv2.waitKey(1)

    def showStats(self, globObjNum, objNum, posAndHght, posCov, poseTypeProb, score, orientation, numColPix):
        
        nonZerosInds = np.where(poseTypeProb > 0.01)[0]
        numRows = nonZerosInds.shape[0] + 1
        
        txtHght = 50
        Margin = 10
        col1 = 125
        col2 = 500
        width = 900

        winHght = numRows * txtHght + 2 * Margin
        img = 255 * np.ones((winHght,width,3), np.uint8)
        
        color = (0,0,0)
        drawThickness = 1
        fontSize = 0.5
        rw = 0
        Text = 'Frame num: {:d}, Glob ID: {:d}, Local ID: {:d}, Score: {:.2f}, numColPix: {:d}'.format(self.frameNum, globObjNum, objNum, score, int(numColPix))
        x = Margin
        y = Margin + int((rw + 0.5) * txtHght)

        cv2.putText(img,Text,(x,y), cv2.FONT_ITALIC, fontSize,color,drawThickness)

        for rw, i in enumerate(nonZerosInds):
            x = Margin
            y = Margin + int((rw + 1.5) * txtHght)
            cv2.putText(img,self.posenet_helper.postures[i] ,(x,y), cv2.FONT_ITALIC, fontSize,color,drawThickness)
            x += col1
            Text = 'Prob: {:.2f}, PosAndHght: {:.1f}, {:.1f}, {:.2f}'.format(poseTypeProb[i],posAndHght[i,0],posAndHght[i,1],posAndHght[i,2])
            
            cv2.putText(img,Text,(x,y), cv2.FONT_ITALIC, fontSize,color,drawThickness)
            y -= int(txtHght/6)
            for k in range(3):
                x = col2
                Text = '{:.2f}, {:.2f}, {:.2f}'.format(posCov[i,k,0],posCov[i,k,1],posCov[i,k,2])
                cv2.putText(img,Text,(x,y), cv2.FONT_ITALIC, fontSize,color,drawThickness)
                y += int(txtHght/3)

        cv2.namedWindow('objstats', cv2.WINDOW_NORMAL)

        cv2.resizeWindow('objstats', width,winHght)
        cv2.imshow('objstats', img)
        key = cv2.waitKey(1)

        if 0:
            # show a plot of orientatino also
            plt.close()
            plt.bar(np.arange(orientation.size), orientation)
            plt.title('Orientation prob dist')
            plt.show()
            d = 1

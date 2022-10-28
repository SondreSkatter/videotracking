import cv2
import numpy as np
import time
import posenet_helper
import os, glob
import re
import matplotlib.pyplot as plt
import textwrap 

class GT_labeling_tool:
    def __init__(self, Channels, trckCfg, sceneScope, myCamMappers):
        self.objMoveStepLength = 0.1 # meters
        self.heightStepLength = 0.05

        self.fps = trckCfg["targetPeopleDetectFPS"]
        self.numChannels = len(Channels)
        self.keepAspectRatios = True
        self.myCamMappers = myCamMappers
        self.myChannels = Channels
        self.sceneScope = sceneScope
        self.showPoses = False        

        self.makeVideo = False

        if self.showPoses:
            self.posenet_helper = posenet_helper.posenet_helper()
            self.objRef = [None] * self.numChannels

        # create folders to dump incremental groutd truth files into        
        self.GTbitSubFolders = [None] * self.numChannels
        for i in range(self.numChannels):
            try:
                Folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(Channels[i].CapturePath))))
                self.GTbitSubFolders[i] = Folder + '/GTbitsFolder/'
                os.mkdir(self.GTbitSubFolders[i])
            except:
                print("folder already exists. Proceeding to add to folder: " + self.GTbitSubFolders[i])
        try:
            Folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(Channels[0].CapturePath))))
            self.GTbitFolder = Folder + '/GTbitsFolder'
            os.mkdir(self.GTbitFolder)
        except:
            print("folder already exists. Proceeding to add to folder: " + self.GTbitFolder)

        # determine the layout of combined window
        # Need to deal with potentially different aspect ratios    
        self.numFrames =  self.numChannels+2 # will add one frame for the ground truth tool thingy, and another one for text instructions (shortkeys etc.)
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
            nY = int(np.ceil((self.numFrames) / nX))   
            combinedRatio[j] = nX * self.avAspectRatio / nY
        bestChoice = np.argmin(np.abs(np.log(combinedRatio)-np.log(self.avAspectRatio)))
        self.nX = bestChoice + 1
        self.nY = int(np.ceil(self.numFrames / self.nX))      

        self.allNames = []
        self.nameMappedToTrackingID = np.zeros(0, np.int)
        try:
            with open(self.GTbitFolder+'/groundtruth_names.txt', 'r') as filehandle:
                for line in filehandle:
                    currentPlace = line[:-1]
                    # add item to the list
                    self.allNames.append(currentPlace)
                    self.nameMappedToTrackingID  = np.append(self.nameMappedToTrackingID,-1)
        except:
            print('No names file available.')
            pass

        cv2.namedWindow('cameraview', cv2.WINDOW_NORMAL)

        self.Colors = np.array([[255,0,0],[0,0,255],[0,255,0],[255,128,0],[153,51,255],[255,255,0],[255,0,127],[51,255,153]],int)
        self.imBufferInitialized = False
        self.firstFrameProcessed = False
        self.selectedID = -1

    def close(self):
        self.compileGTinfo()
        cv2.destroyWindow('cameraview')
        if self.makeVideo:
            self.out.release()


    def initBuffers(self, imgBuff):
        # this should be called upon receiving the first images; since the images may have been downscaled
        # we don't necessarily know the size at init time
        channelDims = np.zeros((self.numChannels,2),np.int)
        for i in range(self.numChannels):
            channelDims[i,:] = imgBuff[i].shape[0:2]

        self.avAspectRatio = np.sum(channelDims[:,1]) / np.sum(channelDims[:,0])  # might have changed slightly after the rescale due tot the stride issue in pose est
        self.aspectRatios = channelDims[:,1] / channelDims[:,0]
            
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
            self.colPos[i,1] = leftPosAccum-1
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
                    topPosAccum += int(0.65 * (self.rowPos[i-1,1] - self.rowPos[i-1,0] + 1))
            else:
                topPosAccum += self.cmnImSz[0]
            self.rowPos[i,1] = topPosAccum-1

        # for the GT window, we know the scope self.sceneScope, will flip x and y so it align best with aspect ratios
        self.sceneSize = np.array((self.sceneScope[2]-self.sceneScope[0], self.sceneScope[3]-self.sceneScope[1]))     
        GTaspRatio = self.sceneSize[1] / self.sceneSize[0]
        thisRow, thisCol = self.numChannels // self.nX, self.numChannels % self.nX
        self.GTframeSize = np.array((self.rowPos[thisRow,1] - self.rowPos[thisRow,0] + 1, self.colPos[thisCol,1] - self.colPos[thisCol,0] + 1))   
        thisRow, thisCol = (self.numChannels + 1) // self.nX, (self.numChannels + 1) % self.nX        
        self.helpframeSize = np.array((self.rowPos[thisRow,1] - self.rowPos[thisRow,0] + 1, self.colPos[thisCol,1] - self.colPos[thisCol,0] + 1))                                                              
        
        availableRatio = self.GTframeSize[1] / self.GTframeSize[0]
        if 0: #np.log(GTaspRatio) * np.log(availableRatio) > 0: # same-sided aspect ratio
            self.GTxyOrder = np.array((0,1))
            #self.GTsign = 1
        else:
            self.GTxyOrder = np.array((1,0))
            #self.GTsign = 1
        self.sceneScale = np.min(self.GTframeSize / self.sceneSize[self.GTxyOrder])
        self.GTimBuff = np.zeros((self.GTframeSize[0],self.GTframeSize[1],3),np.uint8)
        self.IndexGTimBuff = np.zeros(self.GTimBuff.shape,np.uint8)

        helpImBuff = 190 * np.ones((self.helpframeSize[0],self.helpframeSize[1],3),np.uint8)
 
        thisCol = (self.numChannels + 1) % self.nX
        thisRow = (self.numChannels + 1) // self.nX

        # determine the size of combined window
        self.imDisplay = np.zeros((topPosAccum,leftPosAccum,3),np.uint8)
        self.fontSize = self.imDisplay.shape[0] / 1080 
        self.drawThickness = int(imgBuff[0].shape[0] / 1080 * 2)
        
        cv2.putText(helpImBuff,'To move a person: select by click then move up (u), down (d), left (l),',(50,50), cv2.FONT_ITALIC, self.fontSize,(0,0,0),int(self.drawThickness * 2))
        cv2.putText(helpImBuff,'right (r). To make taller (+), or shorter (-).',(50,100), cv2.FONT_ITALIC,self.fontSize,(0,0,0),int(self.drawThickness * 2))
        cv2.putText(helpImBuff,'To import position and height from previous labeled frame press CTRL+left click.',(50,150), cv2.FONT_ITALIC,self.fontSize,(0,0,0),int(self.drawThickness * 2))
        cv2.putText(helpImBuff,'To add another person press a.',(50,200), cv2.FONT_ITALIC,self.fontSize,(0,0,0),int(self.drawThickness * 2))
        cv2.putText(helpImBuff,'To change or add name, right click and type, when finished press enter.',(50,250), cv2.FONT_ITALIC,self.fontSize,(0,0,0),int(self.drawThickness * 2))
        cv2.putText(helpImBuff,'To delete someone, just name the del. To abort a naming press ESC.',(50,300), cv2.FONT_ITALIC,self.fontSize,(0,0,0),int(self.drawThickness * 2))
        cv2.putText(helpImBuff,'When youre happy press n to move to next frame. When done press q.',(50,350), cv2.FONT_ITALIC,self.fontSize,(0,0,0),int(self.drawThickness * 2))

        self.imDisplay[np.ix_(np.arange(self.rowPos[thisRow,0],self.rowPos[thisRow,0]+helpImBuff.shape[0]),np.arange(self.colPos[thisCol,0],self.colPos[thisCol,0]+helpImBuff.shape[1]),np.arange(3))] = helpImBuff

        self.locImBuff = [None] * self.numChannels              
        self.imBufferInitialized = True        
        if self.makeVideo:
            targetVideoSize = np.array((1080,1920),np.int) # if ratio different, then have the larger dimension equal the target value
            self.videoHeightPixels = targetVideoSize[0]
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            AspectRatio = self.colPos[-1,1] / self.rowPos[-1,1]
            if AspectRatio * targetVideoSize[0] > targetVideoSize[1]:
                self.videoHeightPixels = targetVideoSize[0]
                self.videoWidthPixels = int(AspectRatio*self.videoHeightPixels)
            else:
                self.videoWidthPixels = targetVideoSize[1]
                self.videoHeightPixels = int(self.videoWidthPixels / AspectRatio)
            self.out = cv2.VideoWriter('GTvideo.mp4',fourcc, self.fps, (self.videoWidthPixels, self.videoHeightPixels))

    def renderFrame(self,fr, trackingResults, detResults, imgBuff,correct_det2D,keepGoing,
            correct_det3D=None,haltFrame=False,GTitemsMissed=None):

        oldGTdataAVailableForFrame = os.path.exists(self.GTbitSubFolders[0]+str(fr)+'GTbitXY.txt')
        if not self.firstFrameProcessed:
            # Either start on frame 0, or start on a frame that already has groudn truth
            assert fr == 0 or oldGTdataAVailableForFrame, "Please start either with frame 0 or with a frame that has already been labeled"

        # It's reusing the interface from tracker_viz_mgr, so the later inputs are not used
        if not self.imBufferInitialized:
            self.initBuffers(imgBuff)
        
        self.fr = fr
        self.trackingPos = trackingResults["Pos"].copy()        
        #self.trackingPosOrig = self.trackingPos.copy()
        self.trackingInds = trackingResults["Inds"].copy()
        self.reincarn = trackingResults["numReincarnations"].copy()

        self.Names = [None] * self.trackingInds.size  # just the names of the current observations
        self.dbInd = -np.ones(self.trackingInds.size,np.int)
        for i in range(self.trackingInds.size):
            dbMapInd = np.where(self.nameMappedToTrackingID == self.trackingInds[i])[0]
            if dbMapInd.size > 0:
                self.Names[i] = self.allNames[dbMapInd[0]]
                self.dbInd[i] = dbMapInd[0]
            else:
                self.Names[i] = str(self.trackingInds[i])+str(chr(97+(self.reincarn[i] % 26)))
        
        if oldGTdataAVailableForFrame:
            # Also, open the ground truth for that frame if available to initialize names and positions
            oldGTData = np.loadtxt(self.GTbitSubFolders[0]+str(fr)+'GTbitXY.txt',delimiter=',')
            if oldGTData.size > 0:
                if oldGTData.ndim == 1:
                    oldGTData = np.expand_dims(oldGTData,axis=0)
                dbInds = oldGTData[:,0].astype(np.int)

                self.Names = [None] * dbInds.size
                for i in range(dbInds.size):
                    self.Names[i] = self.allNames[dbInds[i]]
            
                self.dbInd = dbInds.copy()
                # Tracking ID might be different since tracking may have started at a different frame. Need to rematch those; using Hungarian matching
                # Match based on 
                import graph_helper
                Dists = np.sqrt(np.square(self.trackingPos[:,0] - np.expand_dims(oldGTData[:,2],axis=1)) +
                    np.square(self.trackingPos[:,1] - np.expand_dims(oldGTData[:,3],axis=1)))
                maxDist = 2.5
                matchesDb, matchesTrack = graph_helper.hungarianDists(Dists, maxDist)

                self.nameMappedToTrackingID = -np.ones(len(self.allNames), np.int)
                trackingIndsBak = self.trackingInds.copy()
                self.trackingInds = -np.ones(dbInds.size,np.int)
                for i in range(matchesDb.size):
                    self.nameMappedToTrackingID[dbInds[matchesDb[i]]] = trackingIndsBak[matchesTrack[i]]
                    self.trackingInds[matchesDb[i]] = trackingIndsBak[matchesTrack[i]]
            
                self.trackingPos = oldGTData[:,2:5].copy()
            else:
                oldGTdataAVailableForFrame = False
        else:
            print("No previous ground truth available for frame ",fr)

        # Check if, in the previous frame there was a person not present in the current frame (happens when people are not that visible, i.e. in the edge of frame)
        if self.fr > 0 and self.firstFrameProcessed:
            prevGTData = np.loadtxt(self.GTbitSubFolders[0]+str(fr-1)+'GTbitXY.txt',delimiter=',')
            if prevGTData.ndim == 1:
                prevGTData = np.expand_dims(prevGTData,axis=0)
            prevDbInds = prevGTData[:,0].astype(np.int)
            indsOfInterest = np.where(np.isin(prevDbInds,self.dbInd,invert=True))[0]
            if indsOfInterest.size > 0:
                # Will add this person in current view so that user can edit, not from scratch but with previous name and position
                for i in indsOfInterest:
                    self.Names.append(self.allNames[prevDbInds[i]])
                    self.dbInd = np.append(self.dbInd,prevDbInds[i])
                    self.trackingInds = np.append(self.trackingInds,-1)
                    self.trackingPos = np.vstack((self.trackingPos,prevGTData[i,2:5]))
        
        self.trackingPosOrig = self.trackingPos.copy()
        self.detResults = detResults
        self.imgBuff = imgBuff
        self.selectedID = -1
        self.firstFrameProcessed = True

        self.drawGTstuff()
        if self.makeVideo:
            self.out.write(cv2.resize(self.imDisplay, (0,0), fy=self.videoHeightPixels/self.imDisplay.shape[0], fx=self.videoWidthPixels/self.imDisplay.shape[1],interpolation = cv2.INTER_AREA))
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                return 0
            else:
                return 1

        while True:
            key = cv2.waitKey(1)
            if self.selectedID > -1:
                #if key != -1:
                #    print(chr(key))
                # object positition mayh be adjusted by user      
                if key == ord('u'): #up
                    self.trackingPos[self.selectedID,self.GTxyOrder[1]] -= self.objMoveStepLength
                    self.drawGTstuff()
                elif key == ord('d'): #"down"
                    self.trackingPos[self.selectedID,self.GTxyOrder[1]] += self.objMoveStepLength
                    self.drawGTstuff()
                elif key == ord('r'): #right
                    self.trackingPos[self.selectedID,self.GTxyOrder[0]] += self.objMoveStepLength
                    self.drawGTstuff()                
                elif key == ord('l'): #left
                    self.trackingPos[self.selectedID,self.GTxyOrder[0]] -= self.objMoveStepLength
                    self.drawGTstuff()         
                elif key == ord('+'): #taller
                    self.trackingPos[self.selectedID,2] += self.heightStepLength
                    self.drawGTstuff()      
                elif key == ord('-'): #shorter
                    self.trackingPos[self.selectedID,2] -= self.heightStepLength
                    self.drawGTstuff()                       
                elif key == ord('o'): #original position...
                    self.trackingPos[self.selectedID,:] = self.trackingPosOrig[self.selectedID,:] 
                    self.drawGTstuff()  
            if key & 0xFF == ord('n'): 
                self.saveGTinfoForFrame()
                return 1
            elif key & 0xFF == ord('q'):
                self.saveGTinfoForFrame()
                self.compileGTinfo()
                return 0
            elif key & 0xFF == ord('p'):
                return -1
            elif key & 0xFF == ord('a'):
                # add another person not present currently
                # Color can be black, name unknown
                self.trackingPos = np.vstack((self.trackingPos,np.array((0.0,0.0,1.65))))
                self.trackingPosOrig = np.vstack((self.trackingPos,self.trackingPos[-1,:]))
                self.trackingInds = np.append(self.trackingInds,-1)
                self.Names.append("Unknown")
                self.dbInd = np.append(self.dbInd,-1)
                self.drawGTstuff()
            time.sleep(0.01)

    def saveGTinfoForFrame(self):        
        # first: the xy positions
        goodInds = np.where(self.dbInd > -1)[0]
        N = goodInds.size
        GTdata = np.zeros((N,6))
        GTdata[:,1] = self.fr
        GTdata[:,0] = self.dbInd[goodInds] 
        GTdata[:,2:5] = self.trackingPos[goodInds,:] 
        GTdata[:,5] = self.trackingInds[goodInds] 
        np.savetxt(self.GTbitFolder+'/'+str(self.fr)+'GTbitXY.txt', GTdata, delimiter=',',fmt='%d,%d,%.2f,%.2f,%.2f,%d') 

   
    def compileGTinfo(self):        
        allData = np.zeros((0,6))
        Files = glob.glob(self.GTbitSubFolders[0]+'*GTbitXY.txt')
        for fname in Files:         
            newData = np.loadtxt(fname,delimiter=',')
            if newData.shape[0] > 0:
                allData = np.vstack((allData,newData))
        sortInd = np.argsort(allData[:,1])
        np.savetxt(self.GTbitFolder+'/groundtruth_true_xy.txt', allData[sortInd,:], delimiter=',',fmt='%d,%d,%.2f,%.2f,%.2f,%d') 
        # also save names
        with open(self.GTbitFolder+'/groundtruth_names.txt', 'w') as filehandle:
            for listitem in self.allNames:
                filehandle.write('%s\n' % listitem)


    def drawGTstuff(self):
        # Finally, draw the last frame: Ground truth window...
        # We'er drawing true 2D space now. The extent is defined by self.sceneScope 
        self.GTimBuff[:] = 190
        self.IndexGTimBuff[:] = 0

        for c in range(self.numChannels):
            self.locImBuff[c] = self.imgBuff[c].copy() 
        
        personRadius = int(0.4 * self.sceneScale) #meters converted to pixels
        for i in range(self.trackingInds.size):
            # draw a circle/cylinder for each person just using the tracker results as ground truth (for now)
            if self.trackingInds[i] > -1:
                colorInd = self.trackingInds[i] % self.Colors.shape[0]
                color = (int(self.Colors[colorInd,2]),int(self.Colors[colorInd,1]),int(self.Colors[colorInd,0]))
            else:
                color = (0,0,0)
            cv2.circle(self.GTimBuff,(int((self.trackingPos[i,self.GTxyOrder[0]] - self.sceneScope[self.GTxyOrder[1]])*self.sceneScale),
                int((self.trackingPos[i,self.GTxyOrder[1]] - self.sceneScope[self.GTxyOrder[0]])*self.sceneScale)),personRadius,color,thickness=-1)
            if self.selectedID == i:
                    # draw a black line aroud teh circle to indicate selection
                cv2.circle(self.GTimBuff,(int((self.trackingPos[i,self.GTxyOrder[0]] - self.sceneScope[self.GTxyOrder[1]])*self.sceneScale),
                    int((self.trackingPos[i,self.GTxyOrder[1]] - self.sceneScope[self.GTxyOrder[0]])*self.sceneScale)),personRadius,(0,0,0),thickness=3)
            cv2.circle(self.IndexGTimBuff,(int((self.trackingPos[i,self.GTxyOrder[0]] - self.sceneScope[self.GTxyOrder[1]])*self.sceneScale),
                int((self.trackingPos[i,self.GTxyOrder[1]] - self.sceneScope[self.GTxyOrder[0]])*self.sceneScale)),personRadius,(i+1,i+1,i+1),thickness=-1, lineType = 4)

            # draw projection on video frames as well
            for c in range(self.numChannels):                
                Res = self.detResults[c]
                if not 'clipOffset' in Res.keys():
                    # Just for backward compatibility with old movies
                    Res["clipOffset"] = np.zeros(2,np.int)
                scale4 = np.array((Res["output_scale"][0],Res["output_scale"][1],Res["output_scale"][0],Res["output_scale"][1]))
                offset4 = np.append(Res["clipOffset"],Res["clipOffset"])
                Point = np.array((self.trackingPos[i,0],self.trackingPos[i,1],0.0))

                transVec = np.cross(self.myCamMappers[c].trans - Point,np.array((0,0,1)))
                transVec /= np.linalg.norm(transVec)
                halfWidth = 0.45 / 2
                Height = self.trackingPos[i,2]

                Points = np.vstack((Point - transVec*halfWidth,Point + transVec*halfWidth,
                    Point + transVec*halfWidth + np.array((0.0,0.0,Height)),Point - transVec*halfWidth + np.array((0.0,0.0,Height)))).transpose()
                cornersX,cornersY = self.myCamMappers[c].get_pixel_from_world_pos_new(Points)
                Points = np.expand_dims(np.vstack(((cornersX-offset4[1])/scale4[1],(cornersY-offset4[0])/scale4[0])).astype(np.int32).transpose(),axis=1)
                cv2.polylines(self.locImBuff[c],[Points],True,color, self.drawThickness * 2)   
                if self.selectedID == i:
                    # add height so its easier to gage correctness
                    Text = self.Names[i]+", {:.0f}cm".format(100*self.trackingPos[i,2])
                else:
                    Text = self.Names[i]
                cv2.putText(self.locImBuff[c],Text,(int(np.min(Points[:,0,0])),int(np.min(Points[:,0,1]))), cv2.FONT_ITALIC, self.fontSize, color, int(self.drawThickness * 2))                

        for c in range(self.numChannels):
            if self.imgBuff[c] != []:
                if c == 0:
                    # also, write the frame number
                    cv2.putText(self.locImBuff[c],str(self.fr),(int(50*self.imgBuff[c].shape[0]/1080),int((1080-50)*self.imgBuff[c].shape[0]/1080)), cv2.FONT_ITALIC, self.fontSize, (0,255,255), int(self.drawThickness * 2))
                thisCol = c%self.nX
                thisRow = c//self.nX

                self.imDisplay[np.ix_(np.arange(self.rowPos[thisRow,0],self.rowPos[thisRow,0]+self.locImBuff[c].shape[0]),np.arange(self.colPos[thisCol,0],self.colPos[thisCol,0]+self.locImBuff[c].shape[1]),np.arange(3))] = self.locImBuff[c]
                #self.indexImAll[np.ix_(np.arange(self.rowPos[thisRow,0],self.rowPos[thisRow,0]+self.imgBuff[c].shape[0]),np.arange(self.colPos[thisCol,0],self.colPos[thisCol,0]+self.imgBuff[c].shape[1]))] = self.indexIm[c]
  
        # Finally, draw the last frame: Ground truth window...
        thisCol = self.numChannels % self.nX
        thisRow = self.numChannels // self.nX
        self.imDisplay[np.ix_(np.arange(self.rowPos[thisRow,0],self.rowPos[thisRow,0]+self.GTimBuff.shape[0]),np.arange(self.colPos[thisCol,0],self.colPos[thisCol,0]+self.GTimBuff.shape[1]),np.arange(3))] = self.GTimBuff
        cv2.imshow('cameraview', self.imDisplay)
        key = cv2.waitKey(1)                    
        cv2.setMouseCallback("cameraview", self.click)

    def deleteCurrentObject(self, objNum):
        # delete this one...
        self.trackingPos = np.delete(self.trackingPos,objNum,axis=0)
        self.trackingPosOrig = np.delete(self.trackingPosOrig,objNum,axis=0)
        self.trackingInds = np.delete(self.trackingInds,objNum)
        #self.reincarn = np.delete(self.reincarn,objNum)
        self.dbInd = np.delete(self.dbInd,objNum)
        self.Names.pop(objNum)

    def click(self, event, x, y, flags, param):
        col = np.max(np.where(x>self.colPos[:,0])[0])
        row = np.max(np.where(y>self.rowPos[:,0])[0])
        objNum = -1
        ch = col + self.nX * row
        if ch == self.numChannels:
            if y-self.rowPos[row,0] > -1 and y-self.rowPos[row,0] < self.IndexGTimBuff.shape[0] and x-self.colPos[col,0] > -1 and x-self.colPos[col,0] < self.IndexGTimBuff.shape[1]:
                objNum = self.IndexGTimBuff[y-self.rowPos[row,0],x-self.colPos[col,0],0]                
            else:
                print("x y position outside of image. ")

        if objNum > 0:
            objNum -= 1
            
            globObj = self.trackingInds[objNum]
            img = self.imDisplay.copy()
            Text = 'Obj num: {:d}'.format(globObj)
            colorInd = globObj % self.Colors.shape[0]
            color = (int(self.Colors[colorInd,2]),int(self.Colors[colorInd,1]),int(self.Colors[colorInd,0]))
            print(event)
            if event == cv2.EVENT_LBUTTONDOWN:
                # show some stats too  
                self.selectedID = objNum                        
                if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON:
                    # CTRL + Left mouse button; import the position from previous frame GT
                    if self.fr > 0:
                        prevGTData = np.loadtxt(self.GTbitSubFolders[0]+str(self.fr-1)+'GTbitXY.txt',delimiter=',')
                        if prevGTData.ndim == 1:
                            prevGTData = np.expand_dims(prevGTData,axis=0)
                        prevDbInds = prevGTData[:,0].astype(np.int)
                        indOfInterest = np.where(prevDbInds == self.dbInd[objNum])[0]
                        if indOfInterest.size == 1:
                            # Will change the position to match that of previous frame
                            self.trackingPos[objNum,:] = prevGTData[indOfInterest,2:5]
                self.drawGTstuff()  

            elif event == cv2.EVENT_RBUTTONDOWN:
                self.selectedID = objNum    
                self.drawGTstuff() 
                # Change the name of the person...
                newName = ""
                # capture keyboard input, new name
                while True:
                    changed = False
                    img = self.imDisplay.copy()
                    key = cv2.waitKey(1)
                    if (key >= 65 and key <= 90) or (key >= 97 and key <= 122):           
                        # New letter!!
                        newName = newName + chr(key)
                        changed = True
                        print(newName)
                    elif key == 27: #Escape
                        break
                    elif key == 13: # Carriage return
                        if newName == "del":
                            self.deleteCurrentObject(objNum)
                            self.selectedID = -1   
                            self.drawGTstuff()
                        elif len(newName) > 0:
                            self.Names[objNum] = newName
                            # check if this name is already part of database of names
                            try:
                                self.dbInd[objNum] = self.allNames.index(newName)
                                self.nameMappedToTrackingID[self.dbInd[objNum]] = globObj
                            except:
                                # add name to database
                                self.allNames.append(newName)
                                self.nameMappedToTrackingID = np.append(self.nameMappedToTrackingID, globObj)
                                self.dbInd[objNum] = len(self.allNames) - 1
                        break
                    elif key == 8: #"Backspace"
                        if len(newName) > 0:
                            newName = newName[0:(len(newName)-1)]
                            changed = True
                        print(newName) 
                    if changed:
                        img = self.imDisplay.copy()
                        cv2.putText(img,newName,(x,y), cv2.FONT_ITALIC,  1,(0,0,0),4)
                        cv2.imshow('cameraview', img)
                    time.sleep(0.01)
                self.selectedID = objNum    
                self.drawGTstuff()               
        else:
            img = self.imDisplay
            cv2.imshow('cameraview', img)
            key = cv2.waitKey(1)

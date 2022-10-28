import os, cv2, time, camera, numpy as np
import video_channels
import human_detector_pose_est, posenet_helper, pose_analyzer
import scene_manager
import matplotlib.pyplot as plt

class view_detected_poses:
    def __init__(self):

        self.channelName = 'wildtrack1'
        #self.channelName = 'oxford'
        recordingName = []
        self.channelName = 'axis-pole'
        recordingName = 'Mar10-2020-1538'  # on stage and with boxes'
        self.frameNum = 282
        self.detFrameRate = 5.0
        self.minScore = 0.10

        self.myConfig = video_channels.channel(self.channelName, recordingName)
        self.cap = cv2.VideoCapture(self.myConfig.CapturePath)
        r, self.img = self.cap.read()
        self.cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.detectFreq = max(1.0, self.cap_fps / self.detFrameRate)
        self.posenet_helper = posenet_helper.posenet_helper()
        self.myCamera = camera.camera(self.img.shape, self.myConfig.calScale, self.myConfig.calib_path, self.myConfig.useGeometricCenterAsRotAxis)        
        cv2.namedWindow('cameraview', cv2.WINDOW_NORMAL)
        self.showFrame(self.frameNum)
        self.poseSolver = pose_analyzer.poseSolver(self.myConfig.sceneMapPath)


    def showFrame(self, frameNum):
        camFrameNum = int(self.detectFreq * frameNum)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, camFrameNum)
        self.frameNum = frameNum
        r, self.img = self.cap.read()
        self.indexIm = np.zeros(self.img.shape, np.uint8)
         
        # get the poses too
        Path = os.path.dirname(self.myConfig.CapturePath)+'/cached_det_results/'
        detResult = human_detector_pose_est.get_frame(frameNum, Path)
        self.Coords = detResult["keypoint_coords"]
        self.scores = detResult["scores"]
        self.keypoint_scores = detResult["keypoint_scores"]

        self.Colors = np.array([[255,0,0],[0,0,255],[0,255,0],[255,128,0],[153,51,255],[255,255,0],[255,0,127],[51,255,153]],int)
        fontSize = 0.7 * self.img.shape[0] / 1080 
        drawThickness = int(self.img.shape[0] / 1080 * 2)
        for i in range(detResult["num"]):     
            if self.scores[i] > self.minScore:
                colorInd = i % self.Colors.shape[0]
                color = (int(self.Colors[colorInd,2]),int(self.Colors[colorInd,1]),int(self.Colors[colorInd,0]))
                Text = 'Obj: {:d}, {:.2f}'.format(i,self.scores[i])
                cv2.putText(self.img,Text,(int(self.Coords[i,0,1]),int(self.Coords[i,0,0])), cv2.FONT_ITALIC,  fontSize, color, drawThickness)
                posenet_helper.draw_single_skel_and_kp(self.img, self.scores[i], self.keypoint_scores[i,:], self.Coords[i,:,:], color, min_pose_score=0.05, min_part_score=0.105)
                posenet_helper.draw_single_skel_and_kp(self.indexIm, self.scores[i], self.keypoint_scores[i,:], self.Coords[i,:,:], (i+1,i+1,i+1), min_pose_score=0.05, min_part_score=0.105, forIndexImage = True)
        
        # draw frame num too
        cv2.putText(self.img,str(frameNum),(50,50), cv2.FONT_ITALIC,  fontSize, (0,255,0), drawThickness)
        cv2.imshow('cameraview', self.img)
        key = cv2.waitKey(1)
        wSize = cv2.getWindowImageRect('cameraview')
        self.winWidth = wSize[2]
        self.winHeight = wSize[3]
        cv2.setMouseCallback("cameraview", self.click)

    def shutdown(self):
        self.cap.release()
        cv2.destroyWindow('cameraview')
        cv2.destroyWindow('objstats')

    def showStats(self, objNum, posAndHght, posCov, poseTypeProb, score, orientation):
        
        nonZerosInds = np.where(poseTypeProb > 0.01)[0]
        numRows = nonZerosInds.shape[0] + 1
        
        txtHght = 50
        Margin = 10
        col1 = 125
        col2 = 600
        width = 1000

        winHght = numRows * txtHght + 2 * Margin
        img = 255 * np.ones((winHght,width,3), np.uint8)
        
        color = (0,0,0)
        drawThickness = 1
        fontSize = 0.5
        rw = 0
        Text = self.channelName + ', frame num: {:d}, Obj num: {:d}, Score: {:.2f}'.format(self.frameNum, objNum, score)
        x = Margin
        y = Margin + int((rw + 0.5) * txtHght)

        cv2.putText(img,Text,(x,y), cv2.FONT_ITALIC, fontSize,color,drawThickness)

        for rw, i in enumerate(nonZerosInds):
            x = Margin
            y = Margin + int((rw + 1.5) * txtHght)
            
            cv2.putText(img,self.posenet_helper.postures[i] ,(x,y), cv2.FONT_ITALIC, fontSize,color,drawThickness)
            x += col1


            _, elevation = self.poseSolver.elevMap.lookupHeight(posAndHght[i,0],posAndHght[i,1])     

            Text = 'Prob: {:.2f}, PosAndHght: {:.1f}, {:.1f}, {:.2f}, {:.1f}'.format(poseTypeProb[i],posAndHght[i,0],posAndHght[i,1],posAndHght[i,2],elevation)
            
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

    def click(self, event, x, y, flags, param):
        
        try:
            objNum = self.indexIm[y,x,0]
        except:
            print("hmm, x y position outside of image. ")
            return

        if objNum > 0:
            objNum -= 1
            img = self.img.copy()
            Text = 'Obj num: {:d}'.format(objNum)
            colorInd = objNum % self.Colors.shape[0]
            color = (int(self.Colors[colorInd,2]),int(self.Colors[colorInd,1]),int(self.Colors[colorInd,0]))
           
            if event == cv2.EVENT_LBUTTONDOWN:
                # show some stats too            
                BBs = self.poseSolver.getBBfromPose(self.keypoint_scores, self.Coords)    
                heightInPixels = BBs[objNum,2] - BBs[objNum,0]
                _, posAndHght, posCov, poseTypeProb, orientation, _ = self.poseSolver.get_person_coords_from_keypoints(self.Coords[objNum,:,:], self.scores[objNum], self.keypoint_scores[objNum,:], heightInPixels, self.myCamera)
                self.showStats(objNum,posAndHght, posCov, poseTypeProb, self.scores[objNum], orientation)  
            textsize = cv2.getTextSize(Text, cv2.FONT_ITALIC, 1, 4)[0]
            if x + textsize[0] > self.img.shape[1]:
                x -= textsize[0] 
            cv2.putText(img,Text,(x,y), cv2.FONT_ITALIC,  1,color,4)
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                # show the position
                img = self.img.copy()
                xPos, yPos = self.myCamera.getGroundPosition(x,y)
                Text = 'Pos x: {:.2f}, Pos y: {:.2f},Pixel x: {:d}, Pixel y: {:d}'.format(xPos[0],yPos[0],x,y)
                textsize = cv2.getTextSize(Text, cv2.FONT_ITALIC, 1, 4)[0]
                if x + textsize[0] > self.img.shape[1]:
                    x -= textsize[0] 
                cv2.putText(img,Text,(x,y), cv2.FONT_ITALIC,  1,(0,0,255),4)
            else:
                img = self.img
        cv2.imshow('cameraview', img)
        key = cv2.waitKey(1)

Test = view_detected_poses()

while True:
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('n'):
        # go to next frame
        Test.showFrame(Test.frameNum+1)
    elif key & 0xFF == ord('p'):
        # go to previous frame
        Test.showFrame(Test.frameNum-1)



Test.shutdown()
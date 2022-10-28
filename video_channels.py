import numpy as np
import os, datetime

class channel:
    def __init__(self, camName, stream='Live'):
        self.hasGT = False
        self.useRecordedVideo = (stream !='Live')
        self.camName = camName
        self.useGeometricCenterAsRotAxis = True
        self.sceneMapPath = []
        self.calScale = 1.0
        self.clipRegion = None  #
        
        if camName[0:4] == 'axis' or camName[0:7] == 'warrior':
            self.sceneScope = np.array((-16, -5, 10, 10)) 
        if camName == 'axis-wall':
            # close up, usingg HD only
            self.clipRegion = np.array([300,950,1220,3175])   
            self.imSize = np.array([2160,3840,3])
            self.lens_angle_coverage = 90 * np.pi / 180 
            self.Folder = 'camera_calib_tools/images/'+self.camName+"/"
            self.calib_path = self.Folder+self.camName+'calibrationFile.txt'
            self.sceneScope = np.array((-16, -5, 10, 10)) 
            if not self.useRecordedVideo:
                self.CapturePath = 'rtsp://root:security@10.208.28.189/axis-media/media.amp?videocodec=h264'
        elif camName == 'axis-pole':
            # close up, usingg HD only
            self.clipRegion = np.array([550,0,2159,3839])   
            self.imSize = np.array([2160,3840,3])
            self.lens_angle_coverage = 90 * np.pi / 180 
            self.Folder = 'camera_calib_tools/images/'+self.camName+"/"
            self.calib_path = self.Folder+self.camName+'calibrationFile.txt'
            if not self.useRecordedVideo:
                self.CapturePath = 'rtsp://root:security@10.208.28.204/axis-media/media.amp?videocodec=h264'
        elif camName == 'hikvision-thermal':
            self.imSize = np.array([512,640,3])
            self.CapturePath = 'rtsp://admin:Terminator2@192.168.1.64'#:554/Streaming/Original/101/'
            #picPath = 'http://admin:Terminator2@192.168.1.64:80/ISAPI/Streaming/channels/101/picture'
            d = 1
        elif camName == 'warrior-pole':
            # close up, usingg HD only
            self.imSize = np.array([int(1080*self.calScale),int(1920*self.calScale),3])
            self.lens_angle_coverage = 92 * np.pi / 180 
            self.Folder = 'camera_calib_tools/images/'+self.camName+"/"
            self.calib_path = self.Folder+self.camName+'calibrationFile.txt'
            if not self.useRecordedVideo:
                self.CapturePath = 'rtsp://admin:123456@10.208.28.193/out.h264'
                #self.CapturePath = 'http://admin:123456@10.208.28.193/out.h264'

        elif camName == 'warrior-wall':
            # far away, ising 4k resolution but cropped view
            self.imSize = np.array([1440,2560,3])
            self.lens_angle_coverage = 92 * np.pi / 180 * self.imSize[1] / (2*1920)
            self.Folder = 'camera_calib_tools/images/'+self.camName+"/"
            self.calib_path = self.Folder+self.camName+'calibrationFile.txt'
            if not self.useRecordedVideo:
                self.CapturePath = 'rtsp://admin:123456@10.208.28.200/out.h264'
                #self.CapturePath = 'http://admin:123456@10.208.28.200:80/video'
        elif camName == 'oxford':           
            self.Folder = 'sampledata/towncentre/'
            self.calib_path = self.Folder+'TownCentre-calibration.ci.txtinv.txt'
            self.imSize = np.array([1080,1920,3])
            self.hasGT = True
            import towncentreGT
            self.myGT = towncentreGT.towncentreGT(self.Folder + 'TownCentre-groundtruth.top.txt', 25.0)     
            self.CapturePath = self.Folder + 'TownCentreXVID.avi'
        elif camName[0:9] == 'wildtrack':
            camNum = camName[9]
            self.Folder = 'sampledata/wildtrack/Wildtrack_dataset/cam' + camNum + '/'
            self.calib_path = self.Folder + 'calibrationFile.txt'
            self.CapturePath = self.Folder + 'cam' + camNum + '.mp4'
            self.imSize = np.array([1080,1920,3])
            self.useGeometricCenterAsRotAxis = False
            self.sceneMapPath = 'sampledata/wildtrack/Wildtrack_dataset/elevationMap.npz'
            self.sceneScope = np.array((-9.0, -3.02, 26.0, 9.02))
            use2DGT = True
            if use2DGT:
                self.hasGT = True
                import towncentreGT
                self.myGT = towncentreGT.towncentreGT(self.Folder + 'groundtruth.top.txt', 2.0)  
        else:
            raise NameError('Unknown configuration string, Check config.py')

        if self.useRecordedVideo and (camName != 'oxford' and camName[0:9] != 'wildtrack'):
            if stream[-4] == '.':
                self.CapturePath = self.Folder + stream
            else:
                testPath = self.CapturePath = self.Folder + 'sampleFolder/' + stream + "/" + stream 
                if os.path.exists(testPath + '.mp4'):
                    self.CapturePath = testPath + '.mp4'
                elif os.path.exists(testPath + '.avi'):
                    self.CapturePath = testPath + '.avi'
                else:
                    print("No video recording is found... This will not work.")

def get_recording_name(aChannel):
    # Want to make sure all the channels get the same recording name, so this function is only called once for all the channels
    sampleFolder = aChannel.Folder + 'sampleFolder'

    if not os.path.isdir(sampleFolder):
        os.makedirs(sampleFolder)

    Name = datetime.datetime.today().strftime("%b%d-%Y-%H%M")    
    newDir = sampleFolder+"/"+Name
    while os.path.isdir(newDir):
        Name = Name + 'a'
        newDir = sampleFolder+"/"+Name
    return Name
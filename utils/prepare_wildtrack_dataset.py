# takes the wildtrack dataset and converts images, camera calibration & GT to internal formats
import glob, os, cv2
import numpy as np
import xmltodict
import calibrate_camera
import camera
import json

dataPath = 'sampledata/wildtrack/Wildtrack_dataset'
convertVideo = False
convertCalibration = False
convertGroundtruth = True
prepareGT3Dinfo = False

for i in range(7):
    # first: make separate folder for each of the 7 cameras
    subPath = dataPath + '/cam' + str(i+1)
    try:
        os.mkdir(subPath)
    except:
        print("folder already existing. Proceeding to add to folder")
    try:
        os.mkdir(subPath + '/cached_det_results')
    except:
        print("folder already existing. Proceeding to add to folder")
        
    if convertVideo:
        # second: save all the frames as video (so we don't have to change anything in the main code)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        out = cv2.VideoWriter(subPath + '/cam' + str(i+1) + '.mp4', fourcc, 2, (1920,1080))

        for j in range(0,2001,5):
            print(j)
            im = cv2.imread(dataPath+'/Image_subsets/C' + str(i+1) +'/' + '{:08d}'.format(j) + '.png')
            out.write(im)
        out.release()

    if convertCalibration:
        # third: convert camera calibration    
        if i < 4:
            camName = '_CVLab' + str(i+1)
        else:
            camName = '_IDIAP' + str(i-3)
        # first: intrinsic calibration. We're using the undistorted version as out images have been undistorted (only the raw videos would have distortion in them)
        with open(dataPath + '/calibrations/intrinsic_zero/intr' + camName + '.xml') as fd:
            doc = xmltodict.parse(fd.read())
        camMat = doc['opencv_storage']['camera_matrix']['data'].split()
        camParams = np.array((float(camMat[0]), float(camMat[4]), float(camMat[2]), float(camMat[5]), 0.0,0.0,0.0,0.0,0.0 ))
        
        # then the extrinsic cal
        with open(dataPath + '/calibrations/extrinsic/extr' + camName + '.xml') as fd:
            doc = xmltodict.parse(fd.read())
        rvec = np.asarray(doc['opencv_storage']['rvec'].split()).astype(float)
        # need to get from Rodriguez vector to quaternion now
        Norm = np.linalg.norm(rvec)
        theta = Norm
        quat = np.zeros(4,np.float)
        quat[0] = np.cos(theta/2)
        quat[1:4] = np.sin(theta/2) * rvec / Norm
        tvec = np.append(0.0,np.asarray(doc['opencv_storage']['tvec'].split()).astype(float)) / 100.0 # m rather than cm

        # these were in the normal transform direction, i.e. world to camera. We want the opposite...
        quat = camera.quaternion_conj(np.expand_dims(quat,axis=1))
        tvec = camera.point_rotation_by_quaternion(np.expand_dims(-tvec,axis=1),quat)

        calibrate_camera.saveCamCalFile(subPath+'/calibrationFile.txt',camParams, np.zeros(5,np.float), tvec[:,0], quat[:,0])

if convertGroundtruth:
    # lastly, the ground truth. Keep to the format used in the Oxford data. One csv file per camera
    csvFiles = []
    for i in range(7):
        csvFiles.append(open(dataPath + '/cam' + str(i+1) + '/groundtruth.top.txt', 'w'))

    positionFile = open(dataPath + '/groundtruth_true_xy.txt', 'w')

    for j in range(0,2000,5):
        with open(dataPath+'/annotations_positions/' + '{:08d}'.format(j) + '.json', 'r') as f:
            distros_dict = json.load(f)
        frameNum = j // 5
        for k in range(len(distros_dict)):
            personID = distros_dict[k]['personID']
            posID = distros_dict[k]['positionID']
            #xPos = -3.067353387742802 + 0.025290958702792 * float(posID % 480)
            #yPos = -9.102241074732728 + 0.025183121181443 * float(posID // 480)
            xPos = -3.0 + 0.0125 + 0.025 * float(posID % 480)
            yPos = -9.0 + 0.0125 + 0.025 * float(posID // 480)

            # did the analysis in matlab, regarding the single inds vs actual x and y
            # the ground truth appears only to be recorded for a flat central section of the field:
            # -3.014 < x < 9.0618 & -9.056 < y < 26.14
            positionFile.write(str(personID)+','+str(frameNum) + ',' + str(xPos) + ',' + str(yPos) + ',' + str(posID) +'\n')

            for l in range(7):
                if distros_dict[k]['views'][l]['xmin'] > -1:
                    csvFiles[l].write(str(personID)+','+str(frameNum)+',0,1,0,0,0,0,'+str(distros_dict[k]['views'][l]['xmin'])+','+ str(distros_dict[k]['views'][l]['ymin']) + ','+ str(distros_dict[k]['views'][l]['xmax']) +','+ str(distros_dict[k]['views'][l]['ymax']) +'\n')
    
    for i in range(7):
        csvFiles[i].close()
    positionFile.close()

if prepareGT3Dinfo:
    # now, try to make sense of the elevation map by utilizing the ground truth

    Cameras = []

    for l in range(7):
        Cameras.append(camera.camera(np.array([1080,1920,3]), 1.0, dataPath + '/cam' + str(l+1) + '/calibrationFile.txt', False))

    # Build up lots of data points with cam position, unit vector and that 3D position...
    allPositionsFile = open(dataPath + '/allpos.txt', 'w')

    for j in range(0,2000,5):
        with open(dataPath+'/annotations_positions/' + '{:08d}'.format(j) + '.json', 'r') as f:
            distros_dict = json.load(f)
        for k in range(len(distros_dict)):
            #personID = distros_dict[k]['personID']
            posID = distros_dict[k]['positionID']

            for l in range(7):
                if distros_dict[k]['views'][l]['xmin'] > -1:
                    xPix = round((distros_dict[k]['views'][l]['xmin'] + distros_dict[k]['views'][l]['xmax'])/2)
                    yPix = distros_dict[k]['views'][l]['ymax'] 

                    xPos, yPos = Cameras[l].getGroundPosition(xPix,yPix)
                    allPositionsFile.write(str(posID) + "," + str(xPos) + "," + str(yPos) + ','+ str(Cameras[l].trans[0]) + ','+ str(Cameras[l].trans[1]) + ','+ str(Cameras[l].trans[2]) + '\n')
    allPositionsFile.close()
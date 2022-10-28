import calibrate_camera, camera
import numpy as np

calFile = 'C:/Users/CDSFSSkatter/Documents/Projects/videotracking/videotrackingrepo/sampledata/towncentre/TownCentre-calibration.ci.txt'
camParams, trans, quat = calibrate_camera.readCamCalFile(calFile)

quat = camera.quaternion_conj(np.expand_dims(quat,axis=1))
trans = camera.point_rotation_by_quaternion(np.expand_dims(np.append(0.0,-trans),axis=1),quat)

mtrx, dist = calibrate_camera.getMtrxFromParams(camParams)
distInv = calibrate_camera.invertDistortion(np.array([1080,1920]), mtrx, dist)
camParamsClass = camParams[4:].copy()
camParams[4:] = distInv
calibrate_camera.saveCamCalFile(calFile+'inv.txt',camParams,camParamsClass, np.squeeze(trans), np.squeeze(quat))

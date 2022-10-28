import numpy as np
import video_channels
import camera
import calibrate_camera
import select_camera_to_calibrate

def computeError(x2,y2,thetaCam,thetaY,thetaThird, Trans):
    quatX = np.array([np.cos(thetaY/2), np.sin(thetaY/2), 0.0, 0.0])
    quatZ = np.array([np.cos(thetaCam/2), 0.0, 0.0, np.sin(thetaCam/2)])
    #quatY = np.array([np.cos(thetaThird/2), 0.0, np.sin(thetaThird/2), 0.0])
    quatComb = camera.quaternion_mult(np.expand_dims(quatX,axis=1),np.expand_dims(quatZ,axis=1))
    #quatComb = camera.quaternion_mult(quatComb,np.expand_dims(quatY,axis=1))

    if 0:
        unitVector = np.array([x2.flatten(), y2.flatten() ,np.ones(x2.size, np.float)])
        unitVector /= np.linalg.norm(unitVector,axis=0)
        rotMat = camera.quat_to_rot_mat(quatComb)
        unitVector = np.expand_dims(np.matmul(rotMat, unitVector[0:3,:]),axis=1)
    else:
        unitVector = np.array([np.zeros(x2.size, np.float), x2.flatten(), y2.flatten() ,np.ones(x2.size, np.float)])
        unitVector /= np.linalg.norm(unitVector,axis=0)
        unitVector = camera.point_rotation_by_quaternion(unitVector,np.expand_dims(quatComb,axis=1))

    t = -Trans[2] / unitVector[2,:]

    # Trans[0] should be 0 always, Trans[1] is the adjustment to the aperture position

    x = t * unitVector[0,:]
    y = -Trans[1] + t * unitVector[1,:]
    pred_Diags = np.sqrt(np.square(x) + np.square(y))[0] 

    Errors = np.array((pred_Diags[0] - LeftDiag, pred_Diags[2] - RightDiag,pred_Diags[1] - midDiag, np.sqrt(np.square(x[0,3]-x[0,1]) + np.square(y[0,3]-y[0,1])) - midToFar,
        np.sqrt(np.square(x[0,0]-x[0,1]) + np.square(y[0,0]-y[0,1])) - LeftToMiddle,
        np.sqrt(np.square(x[0,2]-x[0,1]) + np.square(y[0,2]-y[0,1])) - RightToMiddle,        
        ))

    Error = np.sum(np.square(Errors))

    Error += 100.0 * np.sum((y < 0).astype(np.float)) 
    Error += 100.0 * (y[0,3] < y[0,1]).astype(np.float) 
    Error += 100.0 * (x[0,0] > x[0,2]).astype(np.float) 
    Error += 100.0 * (x[0,0] > 0.0).astype(np.float) 
    Error += 100.0 * (x[0,2] < 0.0).astype(np.float) 
    return Error

config = video_channels.channel(select_camera_to_calibrate.Camera)

try:
    measDists = np.load(config.Folder+'measuredDistances.npz')
    camHeight = measDists["camHeight"]
    LeftDiag = measDists["LeftDiag"]
    midDiag = measDists["midDiag"]
    RightDiag = measDists["RightDiag"]
    LeftToMiddle = measDists["LeftToMiddle"]
    RightToMiddle = measDists["RightToMiddle"]
    midToFar = measDists["midToFar"]
except:
    assert True, 'Be sure to complete measurements and enter them in 2 & 2b . '

try:
    camParamsIntr = np.load(config.Folder+'camLensParamsInv.npz')
    camParamsIntrClassic = np.load(config.Folder+'camLensParams.npz')
except:
    exit("step 3 needs to be completed!")

try:
    pointPos = np.load(config.Folder+'drawPos.npz')
except:
    exit("step 2 needs to be completed!")


# compute the angles
x1 = (pointPos['drawPos'][:,0] - camParamsIntr['cx']) / camParamsIntr['fx']
y1 = (pointPos['drawPos'][:,1] - camParamsIntr['cy']) / camParamsIntr['fy']

r2 = x1*x1 + y1*y1

rFac = 1 + camParamsIntr['k1']*r2 + camParamsIntr['k2']*r2*r2 + camParamsIntr['k3']*r2*r2*r2
# tangential distortion

x2 = x1*(rFac + 2*camParamsIntr['p1']*x1*y1 + camParamsIntr['p2']*(r2 + 2*x1*x1))
y2 = y1*(rFac + 2*camParamsIntr['p2']*x1*y1 + camParamsIntr['p1']*(r2 + 2*y1*y1))

# subtract the geometric center now...
x2 -= x2[-1]
y2 -= y2[-1]

Trans = np.array([0.0, 0.0, camHeight])
thetaCam = 0.0
thetaMain =-np.pi/2 - 0.2
thetaThird = 0.0

Means = np.array([thetaCam, thetaMain, thetaThird, 0.0, 0.0, Trans[2]])
Range = np.array([0.5, 0.15, 1E-10, 1E-10, 1E-10, 1E-10])
Range = np.array([0.5, 0.25, 1E-10, 1E-10, np.square(0.10), np.square(0.02)])

Covs = np.diag(np.square(Range))
# Actually, we will allow the pinhole (aperture) of the camera to be unknown along the camera axis

numTrials = 25000

numKeepers = 100
numSections = int(round(numTrials/(numKeepers*20)))

numTrials = int(np.floor(numTrials / numSections) * numSections)
Error = np.zeros(numTrials, np.float)
nInSec = int(numTrials / numSections)
camParams = np.expand_dims(Means,axis=1) * np.ones(numTrials, np.float)

for s in range(numSections):
    inds = nInSec*s +  np.arange(nInSec)
    if (s > 0):
        # can we do something more? Using Error as weight perhaps??
        prevInds = np.arange(inds[0])
        prevInds = prevInds[np.logical_not(np.isnan(Error[prevInds]))]
        weights = 1.0 / np.square(Error[prevInds] + 0.01)
            
        weights = 1.0 / (Error[prevInds] + 0.01)

        sortInd = np.argsort(weights)
        weights[sortInd[0:(weights.size-numKeepers)]] = 0.0
        weights /= np.sum(weights)        
        Params = camParams[:,prevInds]
        if len(Params.shape) == 1:
            Params = np.expand_dims(Params,axis=0)

        minInd = np.argmin(Error[prevInds])
        if 1:
            Means = np.sum(Params * weights, axis=1)            
        else:            
            Means = camParams[:,minInd]
        Res = Params - np.expand_dims(Means,axis=1)
        Covs = np.matmul(Res * weights, Res.transpose())
        bestError = Error[minInd]
        d = 1
        
    if len(Covs.shape) == 1:
        Covs = np.expand_dims(Covs,axis=1)
    camParams[:,inds]  = np.random.multivariate_normal(Means, Covs, nInSec).transpose()
    
        
    for i in inds:
        #Trans2 = Trans.copy()
        Trans2 = camParams[3:,i].copy()
        Error[i] = computeError(x2,y2,camParams[0,i],camParams[1,i],camParams[2,i], Trans2) 
    print("Stage ",s," completed. Average error for new points: ",np.mean(Error[inds]))

minInd = np.argmin(Error)
minErr = Error[minInd]
bestParams = camParams[:,minInd]
MeanParams = Means

thetaCam = bestParams[0]
thetaMain = bestParams[1]
thetaThird = bestParams[2]

ErrorTest = computeError(x2,y2,thetaCam,thetaMain, thetaThird, bestParams[3:])

# now we want to combine the rotations by a quaternions
# swapping coordinate systems now... z is up, down, y is horizontal down the camera axis

quatX = np.array([np.cos(thetaMain/2), np.sin(thetaMain/2), 0.0, 0.0])
quatZ = np.array([np.cos(thetaCam/2), 0.0, 0.0, np.sin(thetaCam/2)])
quatComb = camera.quaternion_mult(np.expand_dims(quatX,axis=1),np.expand_dims(quatZ,axis=1))

rotationW = quatComb[0,0]
rotationX = quatComb[1,0]
rotationY = quatComb[2,0]
rotationZ = quatComb[3,0]

translationX = 0.0
translationY = 0.0 # even though we calibrated for this one, it will be reset in step 5. It was only needed to correct teh measurements

translationZ = bestParams[5]

camPars = np.array([camParamsIntr['fx'], camParamsIntr['fy'], camParamsIntr['cx'], camParamsIntr['cy'], camParamsIntr['k1'], camParamsIntr['k2'], camParamsIntr['p1'], camParamsIntr['p2'], camParamsIntr['k3']])

camParsClass = np.array([camParamsIntrClassic['k1'], camParamsIntrClassic['k2'], camParamsIntrClassic['p1'], camParamsIntrClassic['p2'], camParamsIntrClassic['k3']])

calibrate_camera.saveCamCalFile(config.calib_path+"preXYref.txt", camPars, camParsClass, np.array([translationX,translationY,translationZ]), quatComb[:,0])

print("Looking good! Move on to step 5!")
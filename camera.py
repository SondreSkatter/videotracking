import numpy as np
import cv2, math

def quaternion_mult(q,r):
    return np.array([r[0,:]*q[0,:]-r[1,:]*q[1,:]-r[2,:]*q[2,:]-r[3,:]*q[3,:],
            r[0,:]*q[1,:]+r[1,:]*q[0,:]-r[2,:]*q[3,:]+r[3,:]*q[2,:],
            r[0,:]*q[2,:]+r[1,:]*q[3,:]+r[2,:]*q[0,:]-r[3,:]*q[1,:],
            r[0,:]*q[3,:]-r[1,:]*q[2,:]+r[2,:]*q[1,:]+r[3,:]*q[0,:]])

def quaternion_conj(q):
    return np.array([[q[0,0]],[-q[1,0]],[-q[2,0]],[-q[3,0]]])

def point_rotation_by_quaternion(points,q):
    return quaternion_mult(quaternion_mult(q,points),quaternion_conj(q))[1:]

def quat_to_rot_mat(q):
    # assume q is normalized
    assert abs(np.linalg.norm(q)-1.0) < 0.00001, "Rotation quaternion must be normalized..."
    rotMat = np.zeros((3,3),np.float)
    rotMat[0,0] = 1.0 - 2.0 * (q[2]*q[2] + q[3]*q[3])
    rotMat[0,1] = 2.0 * (q[1]*q[2] - q[3]*q[0])
    rotMat[0,2] = 2.0 * (q[1]*q[3] + q[2]*q[0])
    rotMat[1,0] = 2.0 * (q[1]*q[2] + q[3]*q[0])
    rotMat[1,1] = 1.0 - 2.0 * (q[1]*q[1] + q[3]*q[3])
    rotMat[1,2] = 2.0 * (q[2]*q[3] - q[1]*q[0])
    rotMat[2,0] = 2.0 * (q[1]*q[3] - q[2]*q[0])
    rotMat[2,1] = 2.0 * (q[2]*q[3] + q[1]*q[0])
    rotMat[2,2] = 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2])
    return rotMat

class camera:
    def __init__(self, imSize, calScale, calFile, useGeometricCenterAsRotAxis):
        f = open(calFile, 'r')
        x = f.readlines()
        f.close()
        self.fx = float(x[0].split(' = ')[1]) * calScale
        self.fy = float(x[1].split(' = ')[1]) * calScale
        self.cx = float(x[2].split(' = ')[1]) * calScale
        self.cy = float(x[3].split(' = ')[1]) * calScale

        translationx = float(x[5].split(' = ')[1]) 
        translationy = float(x[6].split(' = ')[1]) 
        translationz = float(x[7].split(' = ')[1]) 

        self.trans = np.array((translationx,translationy,translationz))
        self.transExpanded = np.expand_dims(self.trans,axis=1)
        # actually, we'll use quaternions...
        rotx = float(x[8].split(' = ')[1])
        roty = float(x[9].split(' = ')[1])
        rotz = float(x[10].split(' = ')[1])
        rotw = float(x[11].split(' = ')[1])

        self.quat = np.array((rotw,rotx,roty,rotz))
        self.rotMat = quat_to_rot_mat(self.quat)

        self.k1 = float(x[12].split(' = ')[1])
        self.k2 = float(x[13].split(' = ')[1])
        self.k3 = float(x[14].split(' = ')[1])
        self.p1 = float(x[15].split(' = ')[1])
        self.p2 = float(x[16].split(' = ')[1])  
        # the following parameters are the classic distortion paramerrs, i.e. from world coordinates to pixels
        self.k1class = float(x[17].split(' = ')[1])
        self.k2class = float(x[18].split(' = ')[1])
        self.k3class = float(x[19].split(' = ')[1])
        self.p1class = float(x[20].split(' = ')[1])
        self.p2class = float(x[21].split(' = ')[1])  
        
        self.k4 = 0.0
        self.n2 = imSize[0]
        self.n1 = imSize[1]
        
        # left, straight, right, down, up, corners(4) 
        # list pairs of x,y inds
        indsOfInterest = np.array(((0, self.n2/2), (self.n1/2, self.n2/2), (self.n1-1, self.n2/2), (self.n1/2, self.n2-1), (self.n1/2,0), \
            (0,0), (0,self.n2-1), (self.n1-1,self.n2-1), (self.n1-1,0)))
        # compute max rad also
        cornerRads2 = np.sum(np.square((indsOfInterest[5:9,:] - np.array((self.cx,self.cy))) / np.array((self.fx,self.fy))),axis=1)
        self.maxRad2 = np.max(cornerRads2)
        if 0:
            self.numRadFunBins = 250
            binWidth = self.maxRad2 / self.numRadFunBins
            rad2Bins = np.linspace(binWidth/2,self.maxRad2-binWidth/2, self.numRadFunBins)
            self.rad2fun = 1 + self.k1*rad2Bins + self.k2*rad2Bins*rad2Bins + self.k3*rad2Bins*rad2Bins*rad2Bins + self.k4*rad2Bins*rad2Bins*rad2Bins*rad2Bins


        if useGeometricCenterAsRotAxis:
            self.get_unit_vectors(np.array([[self.n1/2,self.n2/2]]), True)
        else:
            self.get_unit_vectors(np.array([[self.cx, self.cy]]), True)

        uVec = self.get_unit_vectors(indsOfInterest, False, True)
        viewDirLeft = uVec[0:2,0]
        viewDirLeft /= np.linalg.norm(viewDirLeft)
        viewDirRight = uVec[0:2,2]
        viewDirRight /= np.linalg.norm(viewDirRight)
        viewDirStraight = uVec[0:2,1]
        viewDirStraight /= np.linalg.norm(viewDirStraight)
        self.topTan = uVec[2,4] 
        self.topTan = self.topTan / np.sqrt(1-np.square(self.topTan))
        self.btmTan = uVec[2,3] 
        self.btmTan = self.btmTan / np.sqrt(1-np.square(self.btmTan))
        self.horRotMat = np.array([[viewDirStraight[1],-viewDirStraight[0]],[viewDirStraight[0],viewDirStraight[1]]])
        viewDirLeft = np.matmul(self.horRotMat,np.expand_dims(viewDirLeft,axis=1))
        viewDirRight = np.matmul(self.horRotMat,np.expand_dims(viewDirRight,axis=1))
        self.leftTan = viewDirLeft[0,0] / viewDirLeft[1,0]
        self.rightTan = viewDirRight[0,0] / viewDirRight[1,0]
        # adding one more thing here: The max angle (it pertains to radial distortion)
        uVecCorners = uVec[:,5:9]
        uVecCorners = point_rotation_by_quaternion(np.vstack((np.zeros(4,np.float),uVecCorners)),quaternion_conj(np.expand_dims(self.quat,axis=1)))
        Rads2 = (np.square(uVecCorners[0,:]) + np.square(uVecCorners[1,:])) / np.square(uVecCorners[2,:])
        self.maxInvRad2 = np.max(Rads2)
        if 0:
            # precompute both radial distortion functions
            binWidth = self.maxInvRad2 / self.numRadFunBins
            rad2Bins = np.linspace(binWidth/2,self.maxInvRad2-binWidth/2, self.numRadFunBins)
            self.invRad2fun = 1 + self.k1class*rad2Bins + self.k2class*rad2Bins*rad2Bins + self.k3class*rad2Bins*rad2Bins*rad2Bins 

    def get_unit_vectors(self,xyPixPair, setCenter = False, setMaxAngles = False):
        xPix = xyPixPair[:,0]
        yPix = xyPixPair[:,1]
        x1 = (xPix - self.cx)/self.fx
        y1 = (yPix - self.cy)/self.fy
        N = xyPixPair.shape[0] 
            
        r2 = np.square(x1) +  np.square(y1)
        r2[r2 > self.maxRad2] = self.maxRad2

        # radial distortion
        #rBin = np.floor(self.numRadFunBins * r2 / (self.maxRad2+0.001)).astype(np.int)
        #rFac = self.rad2fun[rBin] 
        rFac = 1 + self.k1*r2 + self.k2*r2*r2 + self.k3*r2*r2*r2 + self.k4*r2*r2*r2*r2
        # tangential distortion
        x2 = x1*(rFac + 2*self.p1*x1*y1 + self.p2*(r2 + 2*x1*x1))
        y2 = y1*(rFac + 2*self.p2*x1*y1 + self.p1*(r2 + 2*y1*y1))
        # get to floor coordinates
        if setCenter:
            self.x20 = x2
            self.y20 = y2
        else:
            x2 -= self.x20
            y2 -= self.y20

            if setMaxAngles:
                self.x_angle_range = np.array((np.min(x2), np.max(x2)))
                self.y_angle_range = np.array((np.min(y2), np.max(y2)))
            unitVector = np.array([x2, y2, np.ones(N, np.float)])
            unitVector /= np.linalg.norm(unitVector,axis=0)
            unitVector = np.matmul(self.rotMat, unitVector)
            return unitVector

    def get_pixel_from_world_pos_new(self, pos3D):
        # need to go the opposite direction now. 
        # assue Pos is 3xn
        pos3D -= self.transExpanded
        posVector = np.matmul(self.rotMat.transpose(), pos3D)
        x1 = posVector[0,:] / posVector[2,:]
        y1 = posVector[1,:] / posVector[2,:]

        x1 += self.x20[0]
        y1 += self.y20[0]
        r2 = x1*x1 + y1*y1
        # need to limit the distortion to the boundary value in case we are out of the frame
        r2[r2 > self.maxInvRad2] = self.maxInvRad2

        # radial distortion
        #rBin = np.floor(self.numRadFunBins *r2 / (self.maxInvRad2+0.001)).astype(np.int)
        #rFac = self.invRad2fun[rBin] 
        rFac = 1 + self.k1class*r2 + self.k2class*r2*r2 + self.k3class*r2*r2*r2 
        # tangential distortion
        x2 = x1*(rFac + 2*self.p1class*x1*y1 + self.p2class*(r2 + 2*x1*x1))
        y2 = y1*(rFac + 2*self.p2class*x1*y1 + self.p1class*(r2 + 2*y1*y1))
        # now, get the pixels
        x = x2 * self.fx + self.cx
        y = y2 * self.fy + self.cy
        return x,y

    def get_pixel_from_world_pos(self, PosX, PosY, deltaPos = np.zeros(2,np.float)):
        # need to go the opposite direction now. Assume Pos is on the floor
        Pos = np.array([0.0, PosX-self.trans[0], PosY-self.trans[1],-self.trans[2]]) 
        unitVector = np.expand_dims(Pos,axis=1) / np.linalg.norm(Pos)
        unitVector = point_rotation_by_quaternion(unitVector,quaternion_conj(np.expand_dims(self.quat,axis=1)))

        x1 = unitVector[0,:] / unitVector[2,:]
        y1 = unitVector[1,:] / unitVector[2,:]
        x1 += self.x20[0]
        y1 += self.y20[0]
        r2 = x1*x1 + y1*y1

        # radial distortion
        rFac = 1 + self.k1class*r2 + self.k2class*r2*r2 + self.k3class*r2*r2*r2 
        # tangential distortion
        x2 = x1*(rFac + 2*self.p1class*x1*y1 + self.p2class*(r2 + 2*x1*x1))
        y2 = y1*(rFac + 2*self.p2class*x1*y1 + self.p1class*(r2 + 2*y1*y1))
        # now, get the pixels
        x = x2 * self.fx + self.cx
        y = y2 * self.fy + self.cy

        if x < 2 or x > self.n1-3 or y < 2 or y > self.n2-3:
            # we're on the edge or outside the frame of view. Determine on which of the four sides...
            # negative values means outside edge
            x = -1
            y = -1
        return x, y

    def get_bb_from_world_pos(self, PosX, PosY, Height, Width, deltaPos=np.zeros(2,np.float)):
        x,y = self.get_pixel_from_world_pos(PosX, PosY, deltaPos)
        if x == -1 or y == -1:            
            return (0,0,0,0)
        # grab a column of pixels up from this one
        x0 = int(x)
        y0 = int(y)

        Inds = np.expand_dims(np.arange(y0+1),axis=1)
        uVecs = self.get_unit_vectors(np.hstack(x0*np.ones((Inds.size,1)),Inds))
        Heights = self.trans[2] + uVecs[2,:] * np.linalg.norm(np.array([PosX-self.trans[0],PosY-self.trans[1]])) / np.sqrt(1 - np.square(uVecs[2,:]))
        bestY = np.argmin(np.abs(Heights - Height))
        # Then determine the pitch for the width
        x0pos,y0pos = self.getGroundPosition(x0,y0)
        xplus,yplus = self.getGroundPosition(x0+1,y0)
        Pitch = np.linalg.norm(np.array([xplus - x0pos, yplus - y0pos]))

        Box = np.array((int(bestY),int(x-Width/2/Pitch), int(y0), int(x+Width/2/Pitch)))
        return Box

    def getGroundPosition(self, xPix, yPix):
        # assumess xPix and yPix are flat vectors
        uVec = self.get_unit_vectors(np.vstack((xPix,yPix)).transpose(), False, True)
        t = -self.trans[2] / uVec[2,:]
        x = self.trans[0] + (t * uVec[0,:])
        y = self.trans[1] + (t * uVec[1,:])
        return x, y




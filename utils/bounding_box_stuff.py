import numpy as np

def intersection_over_union(boxA,boxB):
    # presumes box is made up of top, left, bottom, right
    iou = 0.0
    interArea = max(0,min(boxA[3],boxB[3]) - max(boxA[1],boxB[1]))
    if (interArea > 0):
        interArea *= max(0,min(boxA[2],boxB[2]) - max(boxA[0],boxB[0]))
        if (interArea > 0):
            boxAArea = (boxA[3] - boxA[1]) * (boxA[2] - boxA[0]) 
            boxBArea = (boxB[3] - boxB[1]) * (boxB[2] - boxB[0]) 
            iou = float(interArea)/ float(boxAArea + boxBArea - interArea)
    return iou


def computeOverlaps(boxesA, boxesB):
    nA = boxesA.shape[0]
    nB = boxesB.shape[0]
    IoUs = np.zeros((nA,nB), np.float)
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
    return IoUs

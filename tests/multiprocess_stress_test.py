import multiprocessing
from custom_queue import Queue
import numpy as np
import time

# The idea is to set up a pipeline with multiple workers in each step
# dummy data and dummy work is dished out at various rates

numWorkersAtLevel = np.array((2,3))
numLevels = numWorkersAtLevel.shape[0]

framesPerSecond = 5.0
numChannels = 7
workItemsPerSecond = framesPerSecond * numChannels

# describe memory and processing payloads

memoryPayload = [(1080,1920,3) , (20,100,3), (20,100,3)]
memoryPayload = [(int(1080*0.6),int(1920*0.6),3) , (20,100,3), (20,100,3)]

secOfWorkPerWorkChunk = [0.03, 0.1]
#imSize = (1080,1920,3)
#imSize = (int(1080*0.6),int(1920*0.6),3)
testDurationSec = 5.0

io_qs = [None] * (numLevels + 1)
worker_pools = [None] * numLevels

def calibrate_work_load():    
    maxTime = 0.2 # just to get to a good measurement
    workAmount = 10 * np.ones(1, np.int)
    workTime = np.zeros(0,np.float)
    while True:
        workTime = np.append(workTime,doWork(workAmount[-1]))         
        if workTime[-1] > maxTime:
            break
        workAmount = np.append(workAmount,workAmount[-1] * 3)
    # do a regression
    unitsPerSec = np.sum(workAmount) / np.sum(workTime) 
    return unitsPerSec

def doWork(workAmount):
    x = 0.0
    start_time = time.time()
    for i in range(workAmount):
        x += 1.0
    return time.time() - start_time

unitsWorkPerSec = calibrate_work_load()

def worker_loop(input_q, output_q, unitsWork, memOutputPayload):
    while True:
        if not input_q.empty():
            frame, upstreamWorkTimes, keepGoing = input_q.get()
            if not keepGoing:
                # this is the message used to stop work
                return
            workTime = doWork(unitsWork)
            upstreamWorkTimes = np.append(upstreamWorkTimes,workTime)
            output_q.put((np.zeros(memOutputPayload,np.float), upstreamWorkTimes, True))
        time.sleep(0.001) 

if __name__ == "__main__":
    # set up the pools
    io_qs[0] = Queue()
    for i in range(numLevels):
        io_qs[i+1] = Queue()
        worker_pools[i] = multiprocessing.Pool(numWorkersAtLevel[i], worker_loop, (io_qs[i] ,io_qs[i+1], int(unitsWorkPerSec * secOfWorkPerWorkChunk[i]), memoryPayload[i+1]))  

    # start working!!    
    lastWorkIssuedAtTime = -1.0
    numPlannedWorkItems = int(np.ceil(testDurationSec * workItemsPerSecond))
    maxExecTime = np.zeros(numPlannedWorkItems,np.float)
    meanExecTime = np.zeros(numPlannedWorkItems,np.float)

    timeCompleted = np.zeros(numPlannedWorkItems,np.float)
    workItemsIssued = 0
    workItemsCompleted = 0
    print("Starting " + str(testDurationSec) + " sec test with " + str(numPlannedWorkItems)+" work items." )
    print("Moving " + str(memoryPayload[0]) + " images around with an added " + str(secOfWorkPerWorkChunk)+" work load." )
    print(str(numChannels) + " channels sampled at " + str(framesPerSecond) + " fps")

    print(str(numLevels) + " levels in pipeline. " + str(numWorkersAtLevel) + " workers per level.\n")

    numWorkersAtLevel = np.array((2,2))
    numLevels = numWorkersAtLevel.shape[0]
    firstTimeDoneIssuingWork = False
    startTime = time.time()
    while workItemsCompleted < numPlannedWorkItems:
        elapsedTime = time.time() - startTime
        if workItemsIssued < numPlannedWorkItems:
            frameNum = np.ceil(elapsedTime * framesPerSecond)
            if workItemsIssued < frameNum * numChannels: # elapsedTime - lastWorkIssuedAtTime > 1.0 / workItemsPerSecond:
                # pile on another frame worth off work items
                for fr in range(numChannels):
                    io_qs[0].put((np.zeros(memoryPayload[0],np.uint8),np.zeros(0,np.float), True))
                    workItemsIssued += 1
        else:
            if not firstTimeDoneIssuingWork:
                print("Done dishing out the work...")
            firstTimeDoneIssuingWork = True
        # pop off any completed work
        if not io_qs[numLevels].empty():
            newFrame, exeTimes, _ = io_qs[numLevels].get()
            maxExecTime[workItemsCompleted] = np.max(exeTimes)
            meanExecTime[workItemsCompleted] = np.mean(exeTimes)
            timeCompleted[workItemsCompleted] = elapsedTime
            workItemsCompleted += 1
    print("Test done, cleaning up...\n")
    for level in range(numLevels-1,-1,-1):
        for i in range(numWorkersAtLevel[level]):
            io_qs[level].put(([],[],False)  )
        time.sleep(0.2)
        worker_pools[level].terminate()

    print(str(testDurationSec) + " sec test completed. " + str(workItemsCompleted)+" of " + str(workItemsIssued) + " work items completed in "+'{:.1f}'.format(timeCompleted[-1])+" seconds." )
    print('Mean & max exec time: {:.3f}, {:.3f}, Overall average time: {:.3f}'.format(np.mean(meanExecTime),np.max(meanExecTime),timeCompleted[-1] / workItemsCompleted) + ", Single thread work times: " + str(secOfWorkPerWorkChunk))
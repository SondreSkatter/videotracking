from face_api_client import FaceAPI
import time
import glob
import cv2
import numpy as np

faceImFiles = glob.glob("faceims/*.jpg")

doAsync = 1
doSync = 1
numReps= 10
forceDetectionOnly =  False  # allows to compare the timing if the face recognition is omitted, and only face detectino is run (& failed)

if doAsync:
        
    import asyncio
    from aiohttp import ClientSession
        
    tasks = []   
    async def hello(url,files):
        async with ClientSession() as session:
            async with session.post(url, data = files) as response:
                response = await response.read()
                print(response)

    loop = asyncio.get_event_loop()
    timeStart = time.time()

    for i in range(numReps):
        im = cv2.imread(faceImFiles[i])
        base_uri = 'https://forge-face-recognition.azurewebsites.net' 
        url = base_uri + '/v1/person/identify/' + str(i)
        _, png_buffer = cv2.imencode('.png', im)
        files = {'file': png_buffer.tobytes()}
        task = asyncio.ensure_future(hello(url,files))
        tasks.append(task)

    loop.run_until_complete(asyncio.wait(tasks))

    print("Time per request (async): ", (time.time()-timeStart)/numReps)

if doSync:    
    timeStart = time.time()
    for i in range(numReps):
        im = cv2.imread(faceImFiles[i])
        if forceDetectionOnly:
            im *= 0
        time0 = time.time()
        Response =  FaceAPI.identify(im, str(i))
        print('request: '+Response['status']+' took: ',time.time()-time0)
    print("Time per request (sync): ", (time.time()-timeStart)/numReps)

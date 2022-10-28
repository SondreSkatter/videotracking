import time
from custom_queue import Queue
import multiprocessing
import cv2
import json
import asyncio
from aiohttp import ClientSession
from face_api_client import FaceAPI 


class Response:
    def __init__(self, response, id, reincarns, response_time):
        self.response = response
        self.id = id
        self.reincarns = reincarns
        self.response_time = response_time

class Message:
    pass

class RequestMessage(Message):
    def __init__(self, request, id, reincarns):
        self.request = request
        self.id = id
        self.reincarns = reincarns

class StopMessage(Message):
    pass

def face_api_request(request):
    if request['type'] == 'identify':
        print('identification submitted')
        return FaceAPI.identify(request['image'], request['temp_id'])
    elif request['type'] == 'confirm_and_update':
        print('confirm and update submitted')
        return FaceAPI.confirm_and_update(request['image'], request['db_id'])
    else:
        raise Exception('invalid request')

def face_api_request_worker(input_queue, output_queue):
    while True:
        if input_queue.empty():
            time.sleep(0.01)
        else:
            msg = input_queue.get()
            if not isinstance(msg, Message):
                raise Exception('invalid face request worker message received: not an instance of Message')
            elif isinstance(msg, RequestMessage):
                start_time = time.time()
                response = face_api_request(msg.request)
                response_time = time.time() - start_time
                output_queue.put(Response(response, msg.id, msg.reincarns, response_time))
            elif isinstance(msg, StopMessage):
                return
            else:
                raise Exception('invalid face request worker message received: unexpectred message type')

async def face_request(url,files,output_queue,id, reincarns):
    async with ClientSession() as session:
        async with session.post(url, data = files) as response:
            startTime = time.time()
            response = await response.read()
            output_queue.put(Response(json.loads(response), id, reincarns, time.time()-startTime))


def face_api_request_worker_async(input_queue, output_queue):

    base_uri = 'https://forge-face-recognition.azurewebsites.net' 

    while True:
        loop = asyncio.get_event_loop()
        tasks = []
       
        if input_queue.empty():
            time.sleep(0.01)
        else:
            while not input_queue.empty():
                msg = input_queue.get()
                if not isinstance(msg, Message):
                    raise Exception('invalid face request worker message received: not an instance of Message')
                elif isinstance(msg, RequestMessage):
                    url = None
                    if msg.request['type'] == 'identify':
                        url = base_uri + '/v1/person/identify/' + msg.request['temp_id']
                    elif msg.request['type'] == 'confirm_and_update':
                        url = base_uri + '/v1/person/confirm/' + msg.request['db_id']
                    else:
                        raise Exception('invalid request')
                    if not url is None:
                        _, png_buffer = cv2.imencode('.png', msg.request['image'])
                        files = {'file': png_buffer.tobytes()}
                    task = asyncio.ensure_future(face_request(url,files,output_queue,msg.id, msg.reincarns))
                    tasks.append(task)
                elif isinstance(msg, StopMessage):
                    return
                else:
                    raise Exception('invalid face request worker message received: unexpectred message type')
            loop.run_until_complete(asyncio.wait(tasks))
    
class FaceRequestWorkerPool:
    #initialize the face request worker pool
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.input_queue = Queue()
        self.output_queue = Queue()
        face_api_request_worker_async
        self.pool = multiprocessing.Pool(num_processes, face_api_request_worker_async, (self.input_queue, self.output_queue))
        #self.pool = multiprocessing.Pool(num_processes, face_api_request_worker, (self.input_queue, self.output_queue))
        self.running = True
    
    def submit_request(self, request, id, reincarns):
        if not self.running:
            raise Exception('FaceRequestWorkerPool is not running')
        self.input_queue.put(RequestMessage(request, id, reincarns))

    def flush_responses(self):
        responses = []
        while not self.output_queue.empty():
            responses.append(self.output_queue.get())
        return responses

    def shutdown(self, flag):
        if not self.running:
            raise Exception('FaceRequestWorkerPool is not running')

        while not self.input_queue.empty():
            self.input_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()

        for _ in range(self.num_processes):
            self.input_queue.put(StopMessage())
            
        time.sleep(0.1)
        if not flag:
            self.pool.terminate()
            print('# of requests that couldn\'t be processed: ', self.input_queue.qsize())
        else:
            self.pool.close()
        self.pool.join()
        self.running = False
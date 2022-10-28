import redis
import uuid
import json
import numpy as np
from PIL import Image
import base64
import io
import cv2
import time
from face_api import app
from face_api.utils.time_op import time_op

#class for tracking temporary ids on Redis
class FaceJob:
    #set to 10 minutes expiration on the face job
    expiration_seconds = 600
    #it is a key value database, so you need to separate the key and the data
    def __init__(self, id, data):
        self.id = id
        if 'face_vector' in data:
            self.face_vector = data['face_vector']
        else:
            self.face_vector = [0.0] * 128

        if 'images_processed' in data:
            self.images_processed = data['images_processed']
        else:
            self.images_processed = 0

        if 'expires_at' in data:
            self.expires_at = data['expires_at']
        else:
            self.expires_at = int(time.time()) + self.expiration_seconds

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name != 'expires_at':
            self.__dict__['expires_at'] = int(time.time()) + self.expiration_seconds

    def data(self):
        return {
            'images_processed': self.images_processed,
            'face_vector': self.face_vector,
            'expires_at': self.expires_at
        }
    #process image and average the vector
    def process_image(self, image, face_model):
        new_face_vector, score, height = face_model.execute(image)
        if len(new_face_vector) == 0:
            app.logger.info('>>no face identified')
            return 0
        else: 
            app.logger.info('>>successful identification')
            average_face_vector = np.asarray(self.face_vector)
            average_face_vector *=  self.images_processed 
            average_face_vector = np.add(average_face_vector, new_face_vector) 
            average_face_vector /= (self.images_processed + 1)
            self.images_processed += 1
            self.face_vector = average_face_vector.tolist()
            return 1

class JobTracker:
    def __init__(self, redis_host, redis_port):
        self.db = redis.StrictRedis(host=redis_host, port=redis_port, db=0, decode_responses=True)
    
    def __del__(self):
        self.db.close()

    def update(self, job):
        self.db.set(job.id, json.dumps(job.data()))

    def delete(self, job_id):
        self.db.delete(job_id)

    def create(self, id, job_data):
        job = FaceJob(id, job_data)
        self.update(job)
        return job

    def lookup(self, job_id):
        job_content = self.db.get(job_id)
        return FaceJob(job_id, json.loads(job_content)) if job_content else False
    
    def scan(self):
        return self.db.scan_iter()
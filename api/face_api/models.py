from flask import jsonify
from mongoengine import Document
from mongoengine import BinaryField, StringField, ListField, FloatField, DateTimeField, IntField

import numpy as np
from datetime import datetime
import base64

def encode_image(image):
    return base64.b64encode(image).decode('utf-8')

#Face model for the web api. Add future models here. 
class Face(Document):
    num_required_imgs = 3
    name = StringField(max_length=128, default='')
    image = BinaryField(required=True)
    vector = ListField(field=FloatField(), required=True)
    date =  DateTimeField(default=datetime.utcnow)
    number_of_observations = IntField(required=True, default = num_required_imgs, max_value=100) #think about this

    def np_vector(self):
        return np.asarray(self.vector)
    #match the face with the faces from the database
    def match(face_vector):
        face_threshold = 0.55# 0.36 #np.array([0.56, 0.48, 0.42, 0.36]) 
        db_faces = Face.objects()
        best_match = None
        for db_face in db_faces:
            dist = np.linalg.norm(db_face.np_vector() - face_vector)
            if dist < face_threshold:
                if not best_match or dist < best_match[0]:
                    best_match = dist, db_face
        return best_match and best_match[1]

    #if matches return the existing face, else create a new face
    def match_or_record(image, face_vector):
        face = Face.match(face_vector)
        if face == None:
            face = Face(image= image, vector= face_vector, number_of_observations= Face.num_required_imgs)
            face.save()
        return face

    def confirm_image(self, processed_image): 
        face_threshold = 0.56 #threshold for one sample
        Dist = np.linalg.norm(np.asarray(self.vector) - np.asarray(processed_image))
        if Dist < face_threshold:
            return 'True'
        else:
            return 'False'
    
    def update_vector(self, new_face_vector):
        old_weight = self.number_of_observations/(self.number_of_observations+1)
        new_weight = 1 - old_weight
        
        self.vector = (old_weight * np.asarray(self.vector)) + (new_weight * np.asarray(new_face_vector))
        self.vector = list(self.vector)
        self.number_of_observations += 1

    #serialize api query fields
    def serialize(self, requested_fields=None):
        assert self.id != None
        d = {}
        def add(field, f=lambda x: x):
            if (not requested_fields or field in requested_fields) and self[field] != None:
                d[field] = f(self[field])
        d['id'] = str(self.id)
        add('name')
        add('image', encode_image)
        add('vector')
        add('date')
        add('number_of_observations')
        return d
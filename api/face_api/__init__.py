from flask import Flask, render_template, request, jsonify, g
app = Flask(__name__, template_folder='../frontend/src') #this line should stay here
from flask_cors import CORS
from flask_appbuilder import AppBuilder
from flask_appbuilder.security.mongoengine.manager import SecurityManager
from flask_mongoengine import MongoEngine
from mongoengine.queryset.visitor import Q
from apscheduler.scheduler import Scheduler
from flask_redis import FlaskRedis
from jinja2.exceptions import TemplateNotFound

from face_api.job_tracker import JobTracker, FaceJob
from face_api.face_model import FaceModel
from face_api.models import Face
from face_api.utils.time_op import time_op

import cv2
import base64
import json, bson
import numpy as np
import atexit
import time
from datetime import datetime
import logging
from functools import reduce
import operator
import io
from PIL import Image

app.config.from_envvar('APP_CONFIG_FILE')
#TODO how big the image should be? 
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.logger.setLevel(logging.DEBUG)

import os
print(repr(app.config))
print(os.getenv('APP_CONFIG_FILE'))

cors = CORS(app, resources={r"*": {"origins": "*"}})
db = MongoEngine(app)
redis_client = FlaskRedis(app)
cron_job_trigger_minutes = 10

#creating global app environment 
@app.before_request
def load_job_tracker():
    if 'jt' not in g:
        g.jt = JobTracker(redis_host=app.config['REDIS_HOST'], redis_port=app.config['REDIS_PORT'])

@app.teardown_appcontext
def close_job_tracker(e=None):
    if 'jt' in g:
        del g.jt

@app.before_request
def load_face_recognition_model():
    if 'face_model' not in g:
        g.face_model = FaceModel()

@app.teardown_appcontext
def close_face_model(e=None):
    if 'face_model' in g:
        del g.face_model

cron = Scheduler(deamon=True)
cron.start()
atexit.register(lambda: cron.shutdown(wait=False))

#cron job to prune redis every 10 minutes
@cron.interval_schedule(minutes=cron_job_trigger_minutes)
def prune_redis():
    #figure out the app context
    jt = JobTracker(redis_host=app.config['REDIS_HOST'], redis_port=app.config['REDIS_PORT'])
    for key in jt.scan():
        face_job = jt.lookup(key)
        if int(time.time()) > face_job.data()['expires_at']:
            jt.delete(key)

#returning ok and error status codes along with the request to make it easier to work with the API
def ok(data={}):
    data = data.copy()
    data['status'] = 'ok'
    return jsonify(data)

def error(message):
    return jsonify({'status': 'error', 'error': message})

def parameter_is_valid(param):
    return param and param != '0'

def mutually_exclusive_parameters(param1, param2):
    p1 = request.args.get(param1)
    p2 = request.args.get(param2)
    if parameter_is_valid(p1) and parameter_is_valid(p2):
        return error('parameters "%s" and "%s" are mutually exclusive' % (param1, param2))

def format_date(date):
    return datetime.strptime(date, '%Y-%m-%d')

def read_image(data):
    img = Image.open(io.BytesIO(data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

#index route for person
@app.route('/v1/person', methods=['GET'])
def index_person():
    query = Q()
    def register_query(name, f):
        nonlocal query
        arg = request.args.get(name)
        if parameter_is_valid(arg):
            query = query & f(arg)

    #mutually exclusive parameters can't be passed in together
    err = mutually_exclusive_parameters('name', 'missing_name')
    if err:
        return err

    #register query to the main query
    register_query('missing_name', lambda _: Q(name=''))
    register_query('name', lambda name: Q(name=name))
    register_query('created_after', lambda date: Q(date__gte=format_date(date)))
    register_query('created_before', lambda date: Q(date__lte=format_date(date)))

    #get the fields and serialize 
    query_set = Face.objects(query)
    filter_fields = request.args.getlist('field')
    if filter_fields != None and len(filter_fields) != 0:
        query_set = query_set.only(*filter_fields)
    if(len(query_set) > 0):
        faces = [face.serialize(filter_fields) for face in query_set]
        return ok({'data': faces})
    else:
        return ok({'data' : []})

#route for getting person data with id
@app.route('/v1/person/<string:id>', methods=['GET'])
def get_person(id):
    query_set = Face.objects(id=id)
    if(len(query_set) > 0):
        filter_fields = request.args.getlist('field')
        if filter_fields != None and len(filter_fields) != 0:
            query_set = query_set.only(*filter_fields)
        return ok({'data': query_set[0].serialize(filter_fields)})
    else:
        return error('not found') #give 404

#route for updating the person
@app.route('/v1/person/<string:id>', methods=['PUT'])
def update_person(id):
    data = request.get_json() 
    if data == None or not isinstance(data, dict):
        return error('invalid query')
    
    if 'vector' in data:
        return error('cannot directly set face vector')

    query = Face.objects(id=id)
    if 'image' in data:
        face = query[0]
        data['image'] = base64.b64decode(data['image'])
        image = read_image(data['image'])
        face_vector, _, _ = g.face_model.execute(image)
        face.update_vector(face_vector)
        for k,v in data.items():
            face[k] = v
        face.save()
    else:
        query.update(**data)
    return ok()

#TODO: shorten the function, too many nested statements
@app.route('/v1/person/identify/<string:temp_id>', methods=['POST'])
def get_face_id(temp_id):
    def handle_request():
        if 'file' not in request.files:
            return error('missing field')  
        file = request.files['file'].read()
        if file:
            image = read_image(file)
            face_job = time_op('redis_lookup', lambda: g.jt.lookup(temp_id))
            if face_job:
                result = time_op('process_image', lambda: face_job.process_image(image, g.face_model))
                if result:
                    if face_job.images_processed < Face.num_required_imgs:
                        g.jt.update(face_job)
                        return ok({'temp_id' : temp_id})
                    elif face_job.images_processed == Face.num_required_imgs:
                        face = time_op('database_lookup', lambda: Face.match_or_record(image= file, face_vector=face_job.face_vector))
                        # we should probably put a cap on the the number of images processed and delete the job
                        g.jt.delete(face_job.id)  
                        return ok({'id' : str(face.id), 'name' : str(face.name)})
                    else:
                        return error({'message' : 'not supposed to get more than required images'})
                else:
                    return error({'message' : 'no face identified in received image'})
            else:
                face_job = g.jt.create(temp_id, {})
                result = face_job.process_image(image, g.face_model)
                if result:
                    g.jt.update(face_job)
                    return ok({'temp_id' : face_job.id})
                else:
                    return error({'message' : 'no face identified in received image'})    
        else:
            return error({'message': 'invalid file'})  
    return time_op('handle_identify', handle_request)
 
@app.route('/v1/person/confirm/<string:id>', methods=['POST'])
def confirm_face_id(id):
    if 'file' not in request.files:
        return error('missing field')  
    file = request.files['file'].read()
    if file:
        image = read_image(file)
        face_job = FaceJob(id, {})
        result = face_job.process_image(app.logger, image, g.face_model)
        if result: 
            face_objects = Face.objects(id=id)
            if face_objects == None and len(face_objects) == 0:
                return error('face not found')
            face = face_objects[0]
            confirmation = face.confirm_image(processed_image=face_job.face_vector)
            if confirmation:
                face['image'] = image
                face.save()
            return ok({'confirmation': confirmation})
        else:
            return ok({'confirmation': 'False', 'message' : 'no face identified in received image'})
    return error('file error')

@app.route('/v1/person/<string:id>', methods=['DELETE'])
def delete_person(id):
    query_set = Face.objects(id=id)
    if(len(query_set) > 0):
        face = query_set[0]
        face.delete()
        face.save()
        return ok()
    else:
        return error('id not found')

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return '', 404

# THIS NEEDS TO BE AT THE BOTTOM
@app.route('/', methods=['GET'], defaults={'path': 'index.html'})
@app.route('/<path:path>', methods=['GET'])
def catch_all(path):
    try:
        return render_template(path)
    except TemplateNotFound:
        return '', 404
    except Exception as e:
        raise e
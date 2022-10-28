import base64
import requests
import cv2
import json

class FaceAPI():
    base_uri = 'https://forge-face-recognition.azurewebsites.net' 
    #base_uri = 'http://0.0.0.0:8080'
    def identify(face_image, temp_id):
        url = FaceAPI.base_uri + '/v1/person/identify/' + temp_id
        _, png_buffer = cv2.imencode('.png', face_image)
        files = {'file': png_buffer.tobytes()}
        return  json.loads(requests.post(url, files = files).text)
    
    def confirm_and_update(face_image, id):
        url = FaceAPI.base_uri + '/v1/person/confirm/' + id
        _, png_buffer = cv2.imencode('.png', face_image)
        files = {'file': png_buffer.tobytes()}
        return json.loads(requests.post(url, files = files).text) 
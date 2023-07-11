# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 07:18:55 2023
@author: Khaled
"""
import cv2
import os
import numpy as np
from tensorflow import keras 
from keras.models import load_model
from flask import Flask, request

app = Flask(__name__)
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"


faceCascade = cv2.CascadeClassifier(cascPathface)


model=load_model('my_model.h5')
def get_padding_bbox(x1, y1, w1, h1, real_w, real_h,frame):
    dim=(71,71)
    ratio_bbox_and_image=(w1*h1)/(real_h*real_w)
    x = x1 - int((w1) * (1+ratio_bbox_and_image))
    y = y1 - int((h1) * (1+ratio_bbox_and_image))
    w = w1 + int((w1) * (1+ratio_bbox_and_image))
    h = h1 + int((h1) * (1+ratio_bbox_and_image))
    if x < 0: 
        x = 0
    if y < 0:
        y = 0
    if w > real_w:
        w = real_w
    if h > real_h:
        h = real_h
    padding_img = frame[y:y+h, x:x+w] 
    resized_padding_img = (cv2.resize(padding_img, dim, interpolation = cv2.INTER_AREA)).reshape(1,71,71,3)
    return resized_padding_img

def get_ready(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
    
    x,y,w,h=faces[-1]
    real_h=frame.shape[0]
    real_w=frame.shape[1]
    
    prepared_image=get_padding_bbox(x,y,w,h,real_w,real_h,frame)
    return prepared_image

image_name='alooo.jpg'
# image_path='C:\\Users\\mahmo\\Desktop\\fask app\\'
image_path = base_dir = os.getcwd()
@app.route('/upload', methods=['POST'])
def upload():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return "No image found in the request", 400

    image = request.files['image']

    # Save the image to a local directory
    image.save(image_path+image_name)
    frame=cv2.imread(image_name)
    predictable=get_ready(frame)
    prediction=model.predict(predictable)
    prediction=np.argmax(prediction)
    if(prediction==0):
        return "spoofing", 200
    else:
        return "live", 200
    
    



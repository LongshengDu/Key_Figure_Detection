import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import sys
import copy
import random
import detect_face

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

#face detection parameters
minsize = 25 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

print('Creating networks and loading parameters')
gpu_memory_fraction=1.0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './mtcnn_model/')


#start detect
video = cv2.VideoCapture('./test_examples/office.mp4')

while(video.isOpened()):

    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gray.ndim == 2:
        img = to_rgb(gray)

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]

    for face_position in bounding_boxes:
      
        face_position=face_position.astype(int)
                
        cv2.rectangle(frame, (face_position[0], face_position[1]), 
                            (face_position[2], face_position[3]), 
                            (0, 255, 0), 2)

    cv2.imshow('Test',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import tensorflow as tf
from os.path import join as pjoin
import detect_face

def process_video(video_para):
    # Generate output path
    video_name  = os.path.basename( video_para )
    video_path  = os.path.dirname( video_para )
    output_path = pjoin( video_path, video_name.replace('.', '_') )
    imout_path  = pjoin( output_path, 'faces' )

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(imout_path):
        os.mkdir(imout_path)

    # Load video 
    video  = cv2.VideoCapture(video_para)

    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Face detection parameters
    mtcnn_model = 'detection/mtcnn_model/'
    minsize = 25 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    margin = 32 # crop magrin

    # Restore mtcnn model
    print('Creating networks and loading parameters')
    gpu_memory_fraction=1.0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, mtcnn_model)

    # Start detect
    frame_count = 0
    face_count  = 0
    faceDB = []

    while(video.isOpened()):
        # Get frame
        ret, frame = video.read()

        # Convert to 3D gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.ndim == 2:
            w, h = gray.shape
            img = np.empty((w, h, 3), dtype=np.uint8)
            img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = gray

        # Detect faces
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        # Check faces in frame, generate meta data, store face crop
        for face_position in bounding_boxes:
            # Add to face DB
            face_info = [face_count, frame_count, -1]
            face_info.extend(face_position)
            faceDB.append(face_info)

            print face_info

            # Crop and save face to file
            size  = np.asarray(img.shape)[0:2]
            bb    = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(face_position[0] - margin/2, 0)
            bb[1] = np.maximum(face_position[1] - margin/2, 0)
            bb[2] = np.minimum(face_position[2] + margin/2, size[1])
            bb[3] = np.minimum(face_position[3] + margin/2, size[0])
            crop  = img[bb[1] : bb[3], bb[0] : bb[2], : ]

            imcrop = pjoin(imout_path, str(face_count)+'.png')
            cv2.imwrite(imcrop, crop)

            # Count faces
            face_count += 1

        # Count frames
        frame_count += 1

        # DEBUG USE
        if(face_count > 1000):
            break

    # Close video
    video.release()

    # Save video meta data
    video_meta = np.array([width, height, frame_count, face_count], dtype=np.int64)
    np.save(pjoin(output_path, 'video_meta.npy'), video_meta)

    # Save face DB data
    faces_db = np.array(faceDB)
    np.save(pjoin(output_path, 'faces_db.npy'), faces_db)

    # Return DB path
    return output_path


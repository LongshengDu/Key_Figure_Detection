import os
import numpy as np
import tensorflow as tf
from os.path import join as pjoin
from scipy import misc
import facenet

def face_embed(db_path):   
    # Face embedding parameters
    image_size    = 160
    facenet_model = 'embedding/facenet_model/20180402-114759/'

    # Load video info
    video_meta = np.load(pjoin(db_path, 'video_meta.npy'))
    img_num    = video_meta[3]

    # Load face DB
    face_db = []
    for i in range(img_num):
        filename = pjoin(db_path, 'faces/' + str(i) + '.png')
        img = misc.imread(filename)
        if img is not None:
            face_db.append(img)

    # Align and prewhiten facial image
    for i in range(img_num):
        aligned    = misc.imresize(face_db[i], (image_size, image_size), interp='bilinear')
        face_db[i] = facenet.prewhiten(aligned)

    # Start embedding
    emb = []
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load model
            facenet.load_model(facenet_model)
            # Initialize variables
            images_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
            for i in range(img_num):
                # Display progress
                if i%(img_num/10) == 0:
                    print "Progress: %6.2f %%" % (float(100*i) / img_num)
                # Run forward pass
                feed_dict = { images_placeholder: [ face_db[i] ], phase_train_placeholder: False }
                ret = sess.run(embeddings, feed_dict=feed_dict)
                emb.append(ret[0])
            # Save embedding
            np.save(pjoin(db_path, 'face_embedding.npy'), np.array(emb))
            print "Finished: 100.00 %"

    # Return face_embedding path
    return pjoin(db_path, 'face_embedding.npy')


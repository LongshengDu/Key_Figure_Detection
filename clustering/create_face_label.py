import os
import shutil
import numpy as np
from scipy import misc
from os.path import join as pjoin
from sklearn.cluster import DBSCAN

def face_cluster(db_path):
    # Face embedding parameters
    cluster_threshold  = 0.78
    min_cluster_size   = 12

    # Load face embedding
    face_label = pjoin( db_path, 'face_label' )
    face_path  = pjoin( db_path, 'faces' )
    embedding  = np.load( pjoin( db_path, 'face_embedding.npy' ) )
    print embedding.shape

    # Generate distance matrix
    nrof_images = len(embedding)
    matrix = np.zeros((nrof_images, nrof_images))

    for i in range(nrof_images):
        for j in range(nrof_images):
            dist = np.sqrt(np.sum(np.square(np.subtract(embedding[i, :], embedding[j, :]))))
            matrix[i][j] = dist

    db = DBSCAN(eps=cluster_threshold, min_samples=min_cluster_size, metric='precomputed')
    db.fit(matrix)
    labels = db.labels_.tolist()

    # Get number of clusters
    no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print no_clusters

    # Copy to label folder
    if os.path.exists(face_label):
        shutil.rmtree(face_label)
    os.mkdir(face_label)
    for i in range(nrof_images):
        face = pjoin(face_path, str(i)+'.png')
        labeled = pjoin(face_label, str(labels[i]))
        if not os.path.exists(labeled):
            os.mkdir(labeled)
        shutil.copy2(face, labeled)

    # Save clusters label
    label_db = np.array(labels)
    np.save(pjoin(db_path, 'label_db.npy'), label_db)

    # Return DB path
    return db_path

import numpy as np
from os.path import join as pjoin
from pandas import DataFrame

def analyze_data(db_path):
    meta   = np.load( pjoin( db_path, 'video_meta.npy' ) )
    faces  = np.load( pjoin( db_path, 'faces_db.npy' ) )
    labels = np.load( pjoin( db_path, 'label_db.npy' ) )

    facefrm = faces[:, 1]
    frames  = meta[2]
    window  = 10

    no_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    matrix = np.zeros((no_clusters, no_clusters))

    for i in range(window, frames+1):
        x, = np.where((facefrm >= i-window) & (facefrm < i))
        if len(x)>0:
            upper = max(x)
            lower = min(x)
            snap  = set(labels[lower:upper+1])
            for i in snap:
                for j in snap:
                    if i>-1 and j>-1:
                        matrix[i,j] += 1
    
    print DataFrame(matrix)

    np.save(pjoin(db_path, 'network_matrix.npy'), matrix)

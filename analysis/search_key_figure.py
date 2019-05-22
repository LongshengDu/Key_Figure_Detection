import numpy as np
from os.path import join as pjoin
from pandas import DataFrame

def analyze_data(db_path):
    # Load video info and face info
    meta   = np.load( pjoin( db_path, 'video_meta.npy' ) )
    faces  = np.load( pjoin( db_path, 'faces_db.npy' ) )
    labels = np.load( pjoin( db_path, 'label_db.npy' ) )
  
    # Sliding parameter
    window  = 10
    # Get frame count and face frame stamp
    frames  = meta[2]
    facefrm = faces[:, 1]
    
    # Face label count
    no_ppl = len(set(labels)) - (1 if -1 in labels else 0)
    # Adjacency matrix
    matrix = np.zeros((no_ppl, no_ppl))
    
    # Faces appeared within one continuous frame window deemed related
    for i in range(window, frames+1):
        x, = np.where((facefrm >= i-window) & (facefrm < i))
        # Propulate adjacency matrix
        # matrix[i,i] represents face total appearance time
        if len(x) > 0:
            upper = max(x)
            lower = min(x)
            snap  = set(labels[lower:upper+1])
            for i in snap:
                for j in snap:
                    if i>-1 and j>-1:
                        matrix[i,j] += 1
    
    # Save and print adjacency matrix
    np.save(pjoin(db_path, 'network_matrix.npy'), matrix)
    print DataFrame(matrix)

import cv2
import math
import itertools
import colorsys
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

def get_spaced_colors(n):
    H = np.arange(0, 0.99, 0.99/n)
    S = ([1.0, 0.7] * n)[:n]
    V = ([1.0, 0.7] * n)[:n]
    colors = [ hsv2rgb(h, s, v) for h,s,v in zip(H,S,V) ]
    return colors

def get_color_map(n):
    H = np.arange(0, 0.99, 0.99/n)
    S = ([1.0, 0.7] * n)[:n]
    V = ([1.0, 0.7] * n)[:n]
    colors = [ hsv2rgb(h, s, v) for h,s,v in zip(H,S,V) ]
    color_map = []
    for (r, g, b) in colors:
        color_map.append('#%02x%02x%02x' % (b, g, r))

    return color_map

def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix > 20)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)

    a = list(gr.degree(mylabels.keys()))
    a[:] = [(k, x-2) for (k, x) in a]
    print a

    color_map = get_color_map(len(mylabels.keys()))

    nx.draw(gr, node_color = color_map, node_size=1000, labels=mylabels, with_labels=True)
    plt.show()



mylabels = {}
for i in range(no_clusters):
    mylabels[i] = str(i)
show_graph_with_labels(matrix, mylabels)

# ---------------------------------------------------------

faces_db   = np.load('silicon_valley_mp4/faces_db.npy')
video_meta = np.load('silicon_valley_mp4/video_meta.npy')
label_db   = np.load('silicon_valley_mp4/label_db.npy')

frame_width  = video_meta[0]
frame_height = video_meta[1]
face_num     = video_meta[3]

no_clusters  = len(set(label_db)) - (1 if -1 in label_db else 0)
colors = get_spaced_colors(no_clusters)

cap = cv2.VideoCapture('SiliconValley.mp4')
out = cv2.VideoWriter('silicon_valley.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 24, (frame_width,frame_height))

frame_count = 0
face_count = 0


while(cap.isOpened()):
    ret, frame = cap.read()
    while faces_db[face_count][1] == frame_count:
        face_info = faces_db[face_count]
        bb = face_info[3:7].astype(int)
        label = label_db[face_count]
        if label > -1:
            cv2.rectangle(frame, 
                        (bb[0], bb[1]), (bb[2], bb[3]), 
                        colors[label], 2)
            cv2.putText(frame,'%d(%4.4f)'%(label,face_info[2]), 
                        (bb[0]+2, bb[3]-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, colors[label], thickness = 2, lineType = 2)
        face_count += 1
        if face_num == face_count:
            break

    frame_count += 1

    for i in range(no_clusters):
        cv2.rectangle(frame, 
        (10, 10+25*i), (15, 15+25*i), 
        colors[i], 10)

    out.write(frame)
    cv2.imshow('frame',frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or face_num == face_count:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

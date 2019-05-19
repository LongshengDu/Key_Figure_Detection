import os
from detection import construct_face_db
from embedding import extract_image_feature
from clustering import create_face_label
from analysis import search_key_figure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

video_path = os.path.abspath('./test_examples/silicon_valley.mp4')
db_path    = os.path.abspath('./test_examples/silicon_valley_mp4')

construct_face_db.face_detect(video_path)
extract_image_feature.face_embed(db_path)
create_face_label.face_cluster(db_path)
search_key_figure.analyze_data(db_path)

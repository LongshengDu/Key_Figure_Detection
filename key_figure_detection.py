import os
from detection import construct_face_db
from embedding import extract_image_feature
from clustering import create_face_label


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

video_path = os.path.abspath('./test_examples/silicon_valley.mp4')
db_path    = os.path.abspath('./test_examples/silicon_valley_mp4')

construct_face_db.process_video(video_path)
extract_image_feature.face_embedding(db_path)
create_face_label.face_clustering(db_path)
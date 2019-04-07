import os
from detection import construct_face_db
from embedding import extract_image_feature


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

video_path = os.path.abspath('./test_examples/office.mp4')
db_path    = os.path.abspath('./test_examples/office_mp4')

construct_face_db.process_video(video_path)
# extract_image_feature.face_embedding(db_path)


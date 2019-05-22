# Key_Figure_Detection

Key Figure Detection for Multi-Person Videos

Face detection is based on [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).  
Face embedding is based on [Facenet](https://arxiv.org/abs/1503.03832).

## Inspiration

The code was inspired by several articles and projects as follows:

* [OpenCV Face Recognition](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)  
* [Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)  
* [shanren7/real_time_face_recognition](https://github.com/shanren7/real_time_face_recognition)  
* [A comparison of clustering algorithms for face clustering](http://fse.studenttheses.ub.rug.nl/18064/1/Report_research_internship.pdf)  
* [Face Clustering: Representation and Pairwise Constraints](https://arxiv.org/pdf/1706.05067.pdf)  
* [End-to-end Face Detection and Cast Grouping in Movies Using Erdős-Rényi Clustering](https://arxiv.org/pdf/1709.02458.pdf)  
* [Unsupervised Face Recognition in Television News Media](http://cs229.stanford.edu/proj2017/final-reports/5244380.pdf)  

## Credit

* [davidsandberg/facenet](https://github.com/davidsandberg/facenet).

> facenet.py was taken from https://github.com/davidsandberg/facenet/blob/master/src/facenet.py  
> detect_face.py was taken from https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py  


## Setup Environment

```
sudo apt install libpython-dev
sudo apt install python-tk
sudo apt install python-pip
pip install --user -r requirements.txt
```

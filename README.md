# Udacity facial detection  Facial detection using custom made neural network and haar filters

Gola of this project was building a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face.

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition and emotion recognition. 

In the project we designed our ouwn neural network architecture with several convolutional a drop out layers. Due to practical reasons, the compromise had o be taken into account while designing architecture, as the training load was quite heavy.

As seen on the pictures, the detected facial keypoint (<i>left image</i>) can be even used for applying varius facial filters that are found on smartphones applications. <br>
<img src="https://github.com/koles289/Udacity_facial_detection/blob/master/Face_keypoints.png " width="420"> <img src="https://github.com/koles289/Udacity_facial_detection/blob/master/Face_filter.png" width="400">


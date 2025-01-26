
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Now you can use tf.Session and tf.Graph
graph = tf.Graph()
with graph.as_default():
    session = tf.Session()
#from tensorflow import Graph, Session

#import tensorflow as tf



def extract_face(picture, required_size=(224, 224)):
    #print("filename issss",filename)
    #print("orignal iamge",filename.shape)
    model_graph = tf.get_default_graph()
    with model_graph.as_default():
        tf_session = Session()
        with tf_session.as_default():
            detector = MTCNN()
            results = detector.detect_faces(picture)
            #print("results",results)
            if results:
                x1, y1, width, height = results[0]['box']
                x2, y2 = x1 + width, y1 + height
                face = picture[y1-20:y2+20, x1-20:x2+20]
                #print("face shape",face.shape)
                #print("face data type",face.dtype)
                #print("FACE",face)
                
                #print(im.shape)                
                
                if (len(face)>0):
                    im = Image.fromarray((face * 255).astype(np.uint8))
                    
                    im = im.resize(required_size)
                  
                    face_array = asarray(im)
             
                    return face_array
                else:
                    return
            else:
                return picture
  











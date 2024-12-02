from django.shortcuts import render, HttpResponse
from cv_app.models import Image

import numpy as np
from tensorflow import Graph, Session 
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
import tensorflow as tf
import fd
import cv2

#Dimensions 
img_width, img_height = 224,224 
#Create a bottleneck file
top_model_weights_path = 'E://fyp sameer data//model_weights.h5'
# loading up our datasets
train_data_dir = 'E://fyp sameer data//train' 
validation_data_dir = 'E://fyp sameer data//validation' 
  
#Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1. / 255)
path = 'E://maheen fyp data//class 2-eyes forhead sam//WrinkelsOnEyesForhead.2278.jpg'
imag = load_img(path, target_size=(224, 224)) 
imag = img_to_array(imag) 
imag = np.expand_dims(imag, axis=0)
imag /= 255.
bt_prediction = vgg16.predict(imag)


#training data
generator_top = datagen.flow_from_directory( 
   train_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=5, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_train_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
train_data = np.load('E://fyp sameer data//bottleneck_features_train.npy') 
 
# get the class labels for the training data, in the original order 
train_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
train_labels = to_categorical(train_labels, num_classes=num_classes)



def main_page(request):
    """
    View for the main page:
    1) Receive an image uploaded by the user
    2) Perform image processing on the image
    3) Display the image with the result/prediction
    """

    # This if statement will work when user submits the form (presses the upload button)
    if request.method == "POST":
        # Getting image name and image inputted by user
        image_name = request.POST.get("name")
        image = request.FILES.get("image")
        
        # Creating and saving the image in SQLite database
        image_object = Image.objects.create(name=image_name, image=image)
        y = load_img(image, target_size=(img_width,img_height))

        
        #y=cv2.imread(image)
        y = img_to_array(y)
        y=fd.extract_face(y)
        y=y/255
        y=y.reshape(1,img_height, img_width,3)
        bt_prediction = vgg16.predict(y)
        model_graph = tf.get_default_graph()
        with model_graph.as_default():
            tf_session = Session()
            with tf_session.as_default():
                model = Sequential() 
                model.add(Flatten(input_shape=train_data.shape[1:])) 
                model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
                model.add(Dropout(0.5)) 
                model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
                model.add(Dropout(0.3)) 
                model.add(Dense(num_classes, activation='softmax'))
                model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=1e-4),
                   metrics=['acc'])
                model.load_weights(top_model_weights_path)
            
                preds = model.predict_proba(bt_prediction)
                ids=[]
                label=[]
                percenatge=[]
                count=0
                faces = ['NoFace', 'LowWrinkels', 'HighWrinkels', 'Moderate', 'unwrinkled']
                for idx, face, x in zip(range(0,6), faces , preds[0]):
                
                   # print("ID: {}, Label: {} {}%".format(idx, face, round(x*100,2) ))
                    #print("id is ",idx)
                    ids.insert(count,idx)
                    #print("id is ",ids[count])
                    label.insert(count,face)
                    #print("label is ",label[count])
                    percenatge.insert(count,round(x*100,2))
                    #print("result is ",percenatge[count],"%")
                    count==count+1
                    
                a="Id :"+ str(ids[0])+","+" Label is :"+ str(label[0])+","+" Result is :"+ str(percenatge[0])
                b="Id :"+ str(ids[1])+","+" Label is :"+ str(label[1])+","+" Result is :"+ str(percenatge[1])
                c="Id :"+ str(ids[2])+","+" Label is :"+ str(label[2])+","+" Result is :"+ str(percenatge[2])
                d="Id :"+ str(ids[3])+","+" Label is :"+ str(label[3])+","+" Result is :"+ str(percenatge[3])
                e="Id :"+ str(ids[4])+","+" Label is :"+ str(label[4])+","+" Result is :"+ str(percenatge[4])


                
                class_predicted = model.predict_classes(bt_prediction)
                if class_predicted == [0]:
                    image_prediction = "NO face / other object"               
                elif class_predicted == [1]:
                    image_prediction = "Low Wrinkles detected / 25-50%"
                elif class_predicted == [2]:
                    image_prediction = "high Wrinkles detected / 75-100%"                
                elif class_predicted == [3]:
                    image_prediction = "Moderate Wrinkles detected / 50-75%"
                elif class_predicted == [4]:
                    image_prediction = "No wrinkels detected / 0%"                
                else:
                    image_prediction = "no prediction"

                
    
       


        # These variables get passed to our HTML template for displaying
        context = {
        'image_name': image_name,
        'image': image,
        'image_prediction': image_prediction,
        'result1':e,
        'result2':d,
        'result3':c,
        'result4':b,
        'result5':a
        }
    
        return render(request, 'cv_app/main_page.html', context)
    
    # This is what runs when page is normally loaded on a browser (i.e: with a GET request)
    image_prediction = 'No prediction yet'

    context = {
        'image_name': None,
        'image': None,
        'image_prediction': None,
    }
    
    return render(request, 'cv_app/main_page.html', context)


def signin(request):
    return render(request, 'cv_app/signin.html')


def signup(request):
    return render(request, 'cv_app/signup.html')


def Recommendation(request):
    return render(request, 'cv_app/Recommendation.html')

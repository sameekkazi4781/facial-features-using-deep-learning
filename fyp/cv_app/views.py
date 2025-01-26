import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers, applications
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Clear the Keras backend session to avoid conflicts
from keras import backend as K
K.clear_session()

# Dimensions
img_width, img_height = 224, 224

# Paths
top_model_weights_path = 'E://fyp sameer data//model_weights.h5'

# VGG16 Model
vgg16 = applications.VGG16(include_top=False, weights='imagenet')

# Image Data Generator
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training data
train_data_dir = 'E://fyp sameer data//train'
generator_top = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=5,
    class_mode='categorical',
    shuffle=False
)
nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

# Bottleneck features
train_data = np.load('E://fyp sameer data//bottleneck_features_train.npy')
train_labels = to_categorical(generator_top.classes, num_classes=num_classes)

def main_page(request):
    if request.method == "POST":
        # Get the uploaded image
        image_name = request.POST.get("name")
        uploaded_image = request.FILES.get("image")
        
        if not uploaded_image:
            return render(request, 'cv_app/main_page.html', {'error': 'No image uploaded!'})

        # Save the uploaded file temporarily
        temp_image_path = default_storage.save(uploaded_image.name, uploaded_image)
        temp_image_full_path = os.path.join(default_storage.location, temp_image_path)

        try:
            # Load the image
            img = load_img(temp_image_full_path, target_size=(img_width, img_height))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Extract features using VGG16
            bottleneck_features = vgg16.predict(img_array)

            # Load the top model and make predictions
            model = Sequential()
            model.add(Flatten(input_shape=train_data.shape[1:]))
            model.add(Dense(100, activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(50, activation="relu"))
            model.add(Dropout(0.3))
            model.add(Dense(num_classes, activation="softmax"))
            model.load_weights(top_model_weights_path)
            predictions = model.predict(bottleneck_features)
            
            ids = []
            label = []
            percentage = []
            count = 0
            faces = ['NoFace', 'LowWrinkles', 'HighWrinkles', 'Moderate', 'Unwrinkled']

            for idx, face, x in zip(range(0, 6), faces, predictions[0]):
                ids.insert(count, idx)
                label.insert(count, face)
                percentage.insert(count, round(x * 100, 2))
                count += 1

            a = f"Id: {ids[0]}, Label: {label[0]}, Result: {percentage[0]}%"
            b = f"Id: {ids[1]}, Label: {label[1]}, Result: {percentage[1]}%"
            c = f"Id: {ids[2]}, Label: {label[2]}, Result: {percentage[2]}%"
            d = f"Id: {ids[3]}, Label: {label[3]}, Result: {percentage[3]}%"
            e = f"Id: {ids[4]}, Label: {label[4]}, Result: {percentage[4]}%"

            class_predicted = np.argmax(predictions, axis=1)
            if class_predicted == [0]:
                image_prediction = "NO face / other object"
            elif class_predicted == [1]:
                image_prediction = "Low Wrinkles detected / 25-50%"
            elif class_predicted == [2]:
                image_prediction = "High Wrinkles detected / 75-100%"
            elif class_predicted == [3]:
                image_prediction = "Moderate Wrinkles detected / 50-75%"
            elif class_predicted == [4]:
                image_prediction = "No wrinkles detected / 0%"
            else:
                image_prediction = "No prediction"

        except Exception as e:
            return render(request, 'cv_app/main_page.html', {'error': str(e)})

        # These variables get passed to our HTML template for displaying
        context = {
            'image_name': image_name,
            'image': uploaded_image,
            'image_prediction': image_prediction,
            'result1': e,
            'result2': d,
            'result3': c,
            'result4': b,
            'result5': a
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

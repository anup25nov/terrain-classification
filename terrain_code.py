# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import os
import pandas as pd
import skimage.io
import skimage.transform 
from tensorflow import keras


trainDir = 'terrain_classificat_dataset'
class_names = ['bedrock', 'rock', 'sand']
channels = 3
#normalization value should be 255 for rgb or greyscale images.  it should ve 1 for HSV images
normalizationVal = 255.0
print('Loading data')
#converting images to numpy arrays
X = []
y = []
#for train images
for dirname in os.listdir(trainDir):
    if dirname in class_names: 
        classdir = trainDir + '/' + dirname

        for filename in os.listdir(classdir):
            if filename.endswith('.jpg'):
                fnWithPath = classdir + '/' + filename
                image_data = skimage.io.imread(fnWithPath)
                new_image_data = skimage.transform.resize(image_data, (256, 256, channels))
                new_image_data = new_image_data.reshape((256, 256, channels)).astype(np.float32) / normalizationVal
                X.append(new_image_data)
                y.append(dirname)
                
X = np.array(X)
y = np.array(y)

#import label encoder
from sklearn import preprocessing
#label_encoder object knows how to understand word-labels
label_encoder = preprocessing.LabelEncoder()
#Encode labels in colums 'species'
y = label_encoder.fit_transform(y)
print("Splitting the data")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)


#resnet101_model
resnet101_model = keras.applications.ResNet101(weights=None, input_shape=(256,256,3), classes=3)
resnet101_model.compile(optimizer = 'rmsprop',metrics =['accuracy'], loss='sparse_categorical_crossentropy')
resnet101_history = resnet101_model.fit(x = X_train, y=y_train, epochs=50, validation_split = 0.2)


print("Resnet101 model has been finished")


resnet101_model.save("resnet101_model.h5")

#resnet101_trained_model = keras.models.load_model("resnet101_model.h5")
#resnet101_trained_model.summary()
#result = resnet101_trained_model.predict(y_test)

print('Result file creation is in progress')
_accuracy =[]
_val_accuracy =[]
_loss =[]
_val_loss =[]

_loss =resnet101_history.history['loss']
_val_loss =resnet101_history.history['val_loss']

_accuracy = resnet101_history.history['accuracy']
_val_accuracy =resnet101_history.history['val_accuracy']

df =pd.DataFrame(np.array([_accuracy,_val_accuracy,_loss,_val_loss]).T)
df.columns=['Accuraccy', 'Val_accuracy', 'Loss', 'Val_loss']

df.to_csv('resnet101_result.csv')

print("result file is created")


#resnet50_model
resnet50_model = keras.applications.ResNet50(weights = None, input_shape = (256, 256, 3), classes = 3)
resnet50_model.compile(optimizer = 'rmsprop', metrics = ['accuracy'], loss = 'sparse_categorical_crossentropy')
resnet50_history = resnet50_model.fit(x = X_train, y = y_train, epochs = 50, validation_split = 0.2)


print("Resnet50 model has been finished")

print('Result file creation is in progress')
_accuracy = []
_val_accuracy = []
_loss = []
_val_loss = []

_loss = resnet50_history.history['loss']
_val_loss = resnet50_history.history['val_loss']

_accuracy = resnet50_history.history['accuracy']
_val_accuracy = resnet50_history.history['val_accuracy']

df = pd.DataFrame(np.array([_accuracy, _val_accuracy, _loss, _val_loss]).T)
df.columns = ['Accuracy','Val_accuracy', 'Loss', 'Val_loss']

df.to_csv('resnet50_result.csv')


resnet50_model.save("resnet101_model.h5")

#resnet50_trained_model = keras.models.load_model("resnet50_model.h5")
#resnet50_trained_model.summary()
#result = resnet50_trained_model.predict(y_test)

print("result file is created")

#VGG19_model
VGG19_model = keras.applications.VGG19(weights = None, input_shape = (256, 256, 3), classes = 3)
VGG19_model.compile(optimizer = 'rmsprop', metrics = ['accuracy'], loss = 'sparse_categorical_crossentropy')
VGG19_history = VGG19_model.fit(X = x_train, y = y_train, epochs = 50, validation_split = 0.2)

print("VGG19 model has been finished")

print('Result file creation is in progress')
_accuracy = []
_val_accuracy = []
_loss = []
_val_loss = []

_loss = VGG19_history.history['loss']
_val_loss = VGG19_history.history['val_loss']

_accuracy = VGG19_history.history['accuracy']
_val_accuracy = VGG19_history.history['val_accuracy']

df = pd.DataFrame(np.array([_accuracy, _val_accuracy, _loss, _val_loss]).T)
df.columns = ['Accuracy','Val_accuracy', 'Loss', 'Val_loss']

df.to_csv('resnet50_result.csv')

vgg19_model.save("resnet101_model.h5")
#vgg19_trained_model = keras.models.load_model("vgg19_model.h5")
#vgg19_trained_model.summary()
#result = vgg19_trained_model.predict(y_test)
print("result file is created")

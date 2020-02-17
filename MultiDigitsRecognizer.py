
'''
Group Members
1. Muhammad Hamza 1602028 (CS)
2. Ammar Farooq Khan 1602020 (CS)
3. Waseem Haider 1601022 (EE)

'''


# Google Colab Drive mount code
from google.colab import drive
drive.mount('/content/gdrive')



# save the final model to file
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import model_from_json

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# run the these lines for evaluating a model
trainX, trainY, testX, testY = load_dataset()
trainX, testX = prep_pixels(trainX, testX)
model = define_model()
model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
# save model
model_json = model.to_json()
with open("/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/AHW_Hand_Written_Digits_Recognizer.json", "w") as json_file:
	json_file.write(model_json)
model.save('/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/AHW_Hand_Written_Digits_Recognizer.h5')
# evaluate model on test dataset
print('Testing on MNIST test data:')
_, acc = model.evaluate(testX, testY, verbose=0)
print('> %.3f' % (acc * 100.0))

# above code complete the model. Now we can test this model on our dataset...

# Now load the save model completed above...

# load json and create model
json_file = open('/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/AHW_Hand_Written_Digits_Recognizer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/AHW_Hand_Written_Digits_Recognizer.h5")
print("Loaded model from disk")



# Now is time to start the testing...

# Model is done on '99.190' accuracy on MNIST testing dataset
# Now we are testing on our testing dataset
# Our dataset consist of one-to-multiple digits.
# we have a lot Images of single digits, two digits, three digits, Four digits and Five digits on a single image
# First we localize the digits using OpenCV contour() function and some additional own code then pars digits from image
# Finally test them on above train model on multi digits own dataset ...



import cv2
import os
import glob

# Read Images from the specific folder...

labels=[]
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        labels.append(filename)
        if img is not None:
            images.append(img)
    return images


# Uncomment the directory that you want to test...
 # Enter Directory of all images
img_dir = "/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/Multiple_Digits_Dataset/Two_digits"
#img_dir = "/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/Multiple_Digits_Dataset/Three_digits"
#img_dir = "/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/Multiple_Digits_Dataset/Four_digits"
#img_dir = "/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/Multiple_Digits_Dataset/Five_digits"
images=load_images_from_folder(img_dir)

import matplotlib.pyplot as plt
plt.imshow(images[5])
print(len(images)) # Total images in specific folder...
print(labels)     #  Show the Labels of images...



# Image reading end...


#  Now start testing on the above reading multiple digits images from specific folder or directory...
# Libraries...
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np
import cv2
import os

def prep_pixel(test):
  test = test.reshape((test.shape[0], 28, 28,1))
  return test

def sortSecond(val):
    return val[0]

# load json and create model
json_file = open('/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/AHW_Hand_Written_Digits_Recognizer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/AHW_Hand_Written_Digits_Recognizer.h5")
print("Loaded model from disk")

# Model Loaded...

for index in range(len(images)):
  #img = cv2.imread('/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/IMG_20200104_132842.jpg'
  img = images[index]

  # Convert to grayscale and apply Gaussian filtering
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.GaussianBlur(img, (3, 3), 0)
  img=cv2.resize(img,(400,400))
  plt.imshow(img)

  # Refining the image on the based of black and white pixels...
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if(img[i][j]>100 and img[i][j]<255):
        #print(img[i][j])
        img[i][j]=255
      else:
        img[i][j]=0
  #print(img.shape)


  # Threshold the image
  #ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
  im_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
              cv2.THRESH_BINARY_INV, 23, 23)

  # Find contours in the image
  cntr, heir = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  #finding bounding rectangle
  rects = [cv2.boundingRect(con) for con in cntr]

  #print(rects)
  rects.sort(key = sortSecond)
  #print(rects)

  counter = 0
  testImages=list()
  for rect in rects:
    counter = counter + 1
    try:
      leng = int(rect[3] * 1.2)
      pt1 = max(int(rect[1] + rect[3] // 2 - leng // 2), 0)
      pt2 = max(int(rect[0] + rect[2] // 2 - leng // 2), 0)
      roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
      if(roi.shape[0]*roi.shape[1]<100):
        continue
      roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
      img = cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
      # resize the image into (28,28) for model...
      roi=cv2.resize(roi,(28,28))
      roi = roi.astype("float32")/255.0
      plt.imshow(roi)

      testImages.append(roi)

      img_array = img_to_array(testImages)
      #img_array=255-img_array
      #plt.imshow(img_array[0])
      #print(img_array.shape)
      test=prep_pixel(img_array)

      prediction=model.predict_classes(test)
      #print(prediction)

      #nbr = model.predict(roi)
      #print(nbr)
      text = np.argmax(prediction, axis=1)
      cv2.putText(img, str(int(text)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    except Exception as e:
      #print(e)
      pass
  print(prediction,labels[index])
print(counter)
#plt.imshow(img)


# Predictions End and now you can calculate the accuracy...
# We just display the Labels List and comapre it with predicted array...
print(prediction)
print(labels)

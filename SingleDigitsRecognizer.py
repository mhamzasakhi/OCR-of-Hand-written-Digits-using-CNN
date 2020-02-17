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
# Finally test them on above train model on single digits own dataset...


from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np
import cv2


def prep_pixel(test):
  test = test.reshape((test.shape[0], 28, 28,1))
  return test
def refineImage(img):
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if(img[i][j]>100 and img[i][j]<255):
        #print(img[i][j][k])
        img[i][j]=255
      else:
        img[i][j]=0
  return img

imageNames=['zero','one','two','three','four','five','six','seven','eight','nine']
numbers=['1','2','3','4','5','6','7','8','9','10']

labels=list()
prediction=list()
testImages=list()
for i in range(len(imageNames)):
  for j in range(len(numbers)):
    labels.append(int(numbers[i])-1)
    path='/content/gdrive/My Drive/ML3_Hamza_Ammar_Waseem/Single_Digits/'+imageNames[i]+'.'+numbers[j]+'.jpg';
    #print(path)

    img = cv2.imread(path)
    # Convert to grayscale and apply Gaussian filtering
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img=cv2.resize(img,(400,400))
    plt.imshow(img)

    img=refineImage(img)

    #print(img.shape)
    # Threshold the image
    im_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 23, 23)

    # Find contours in the image
    cntr, heir = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #finding bounding rectangle
    rects = [cv2.boundingRect(con) for con in cntr]

    #print(len(rects))

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
        roi = roi.astype("float32")/255
        #plt.imshow(roi)

        testImages.append(roi)
        #nbr = model.predict(roi)
        #print(nbr)
        text = np.argmax(roi, axis=1)
        cv2.putText(img, str(int(text)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
      except Exception as e:
        #print(e)
        pass
    #print(counter)
    #plt.imshow(img)

    img_array = img_to_array(testImages)
    #img_pil = array_to_img(img_array)
    #img_array=255-img_array
    plt.imshow(img_array[0])
    print(img_array.shape)

    test=prep_pixel(img_array)
    #print(test)
    prediction.append(model.predict_classes(test))
for i in range(len(prediction)):
  print(prediction[i],' ',labels[i])


# Calculate the Accuracy and (Correct and Bad) predictions...

Correct=0
bad=0
print(prediction[0][0])
for i in range(len(prediction)):
  if(prediction[i][0]==labels[i]):
    Correct=Correct+1
  else:
    bad=bad+1
print('Correct predictions: ',Correct)
print('Bad predictions: ', bad)
print('Accuracy: ',Correct,'%')

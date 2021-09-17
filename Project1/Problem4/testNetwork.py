from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(file_id='1m5KT8eTf0f_bWZeqT93uVmiLXPH3Gbul', dest_path='./flowers.h5')
gdd.download_file_from_google_drive(file_id='1XR0XmXgpjoJCMZ0MmscPWlETOKAsRd1o', dest_path='./test_image.jpg')

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


def LoadImage(path):
  print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + bcolors.BOLD + " loading and pre-processing image..." + bcolors.ENDC)
  image = load_img(path, target_size=(299, 299))
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = preprocess_input(image)
  return image

def PredictFlowers(model, image):
  print(bcolors.OKGREEN + "[INFO]" + bcolors.ENDC + bcolors.BOLD + " classifying image with '{}'...".format("Inception") + bcolors.ENDC)
  preds = model.predict(image)
  id = np.flip(np.argsort(preds)[0][-3:])
  probs = np.flip(np.sort(preds)[0][-3:])
  i = 0
  for label, prob in zip(id, probs):
    print("{}. {}: ".format(i + 1, CLASSES[label]) + bcolors.BOLD + "{:.2f}%".format(prob * 100) + bcolors.ENDC)
    i += 1


def create_model():
  inception = InceptionV3(input_shape=(299, 299, 3), weights='imagenet', include_top=False)
  inception.trainable = False
  model = tf.keras.Sequential([
    inception,
    GlobalAveragePooling2D(name='global_average_pool'),
    Dropout(0.3, name='dropout1'),
    Dense(1024, activation='relu', name='FC1'),
    Dropout(0.3, name='dropout2'),
    Dense(512, activation='relu', name='FC2'),
    Dropout(0.3, name='dropout3'),
    Dense(5, activation='softmax', name='output')
  ], name='myInception')
  model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
  )
  return model

model = create_model()
model.load_weights('flowers.h5')
image = LoadImage("test_image.jpg")
PredictFlowers(model, image)

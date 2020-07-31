import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import datetime
import keras
import os

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Softmax, Input, Flatten, Conv2D, Activation, BatchNormalization, MaxPooling2D, ReLU, Dropout, Concatenate, Dot, Multiply, Softmax, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
%matplotlib inline

import skimage as sk
from skimage.transform import rescale, resize, rotate
from skimage import util, exposure, io
from numpy import fliplr, flipud

from keras.applications import densenet
# # backend = set_keras_submodules(K.backend()),
from keras_vggface import VGGFace
from tensorflow.python.client import device_lib

from hyperparameters import get_hyperparameters
from load_dataset import preprocess
from load_generator import load_generator
from architecture import single_model
from augmentation import apply_augmentation

attr = pd.read_csv('./dataset/celeba/list_attr_celeba.csv')
eval_partition = pd.read_csv('./dataset/celeba/list_eval_partition.csv')

def load_dataset(attribute_id, num_attributes, generator=True):
    hyperparameters = get_hyperparameters(num_attributes)
    train, val, test = preprocess(hyperparameters, attr, eval_partition,attribute_id)
    if generator:
        train_gen, val_gen, test_gen = load_generator(hyperparameters, train, val, test, attribute_id)
        return hyperparameters, train, val, test, train_gen, val_gen, test_gen
    
    return hyperparameters, train, val, test
 
attribute_id = 0
frac = 0.4
num = apply_augmentation(attribute_id=attribute_id, frac = frac)

lis = os.listdir('./dataset/celeba-dataset/')[:]
l=[]
for i in lis:
    if i[0]=='a':
        l.append(i)
len(l)
z=0
for i in l:
    row1 = attr[attr['image_id'] == i[-10:]]
    row1['image_id'] = i
    attr = attr.append(row1)
    
    row2 = eval_partition[eval_partition['image_id'] == i[-10:]]
    row2['image_id'] = i
    eval_partition = eval_partition.append(row2)
    z+=1
    print(z)
    
attr.reset_index()
eval_partition.reset_index()

hyperparameters, train, val, test, train_gen, val_gen, test_gen = load_dataset(attribute_id = attribute_id, num_attributes=1)

model_vgg = VGGFace(include_top = False, model='vgg16', input_shape=(218, 178, 3), pooling = 'max')
model_vgg.save_weights('saved_history/models/vggface.h5')

x,y,z = hyperparameters['height'], hyperparameters['width'], hyperparameters['channels']
X = Input((x,y,z))
    
tops = single_model(X)  
model = Model(inputs=X, outputs=[tops])
model.load_weights('saved_history/models/model_individual_kd.h5', by_name=True)

opt = SGD(lr = hyperparameters['lr'])    
model.compile(loss='binary_crossentropy',optimizer=opt, metrics = ['accuracy'])
 
print(device_lib.list_local_devices())

date = datetime.datetime.now().strftime("%d - %b - %y - %H:%M:%S")

filename = "vgg_model_individual_"+str(attribute_id = 0)
logdir = "logs/" + filename

filepath = "saved_history/models/model_individual_"+str(attribute_id)
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='max', period = 1)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 1, mode = 'auto', min_lr = 0.00001)

csv_logger = CSVLogger('saved_history/training_results_model_individual_'+str(attribute_id)+ '.csv', separator = ',', append=True)
tensorboard_callback = TensorBoard(log_dir = logdir)

def scheduler(epoch, lr):
    if epoch%3==0:
        print(type(lr))
        return np.float64(lr/10)
    
lr_scheduler = LearningRateScheduler(scheduler)

history = model.fit_generator(train_gen,epochs=3,validation_data = val_gen, callbacks = [lr, csv_logger, tensorboard_callback], verbose = 1)

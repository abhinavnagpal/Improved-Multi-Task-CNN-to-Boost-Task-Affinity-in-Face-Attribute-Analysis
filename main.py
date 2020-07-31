import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import tensorflow

from tensorflow.compat.v1.keras import backend as K
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Softmax, Input, Flatten, Conv2D, BatchNormalization, MaxPooling2D, ReLU, Dropout, Concatenate, Dot, Multiply, Softmax
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model, multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger, TensorBoard, LearningRateScheduler

from load_dataset import preprocess
from load_generator import load_generator
from architecture import single_model
import crossstitch
from hyperparameters import get_hyperparameters

import keras
from keras.applications import densenet
from keras_vggface import VGGFace

class CalculatingPredictions(tf.keras.callbacks.Callback):
    
    def __init__(self, preds, filepath):
        self.preds = preds
        self.filepath = filepath
        
    def on_epoch_end(self, epoch, logs=None):
        predict=model.evaluate(test_gen)
        print(predict)
        self.preds.append(predict)
        k = np.array(preds)
        np.save("saved_history/predictions/predictions_"+ str(epoch)+ ".npy", k)
        date = datetime.datetime.now().strftime("%d - %b - %y - %H:%M:%S")
        model.save_weights(self.filepath+"_epoch_"+ str(epoch) +"_"+date+'.h5')

def main(hyperparameters):
    
    X, tops = architecture(hyperparameters)
    
    # define
    model = Model(inputs=X, outputs=[tops])
    
    #optimizer
    opt = SGD(lr = hyperparameters['lr'])    
    
    #compile 
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics = ['accuracy'])
  
    return model, True

device_name = tf.test.gpu_device_name()
print(device_name)

attr = pd.read_csv('./dataset/celeba/list_attr_celeba.csv')
eval_partition = pd.read_csv('./dataset/celeba/list_eval_partition.csv')
hyperparameters = get_hyperparameters()

train, val, test = preprocess(hyperparameters, attr, eval_partition)
train_gen, val_gen, test_gen = load_generator(hyperparameters, train, val, test)

model_vgg = VGGFace(include_top = False, model='vgg16', input_shape=(160, 128, 3), pooling = 'max')
model_vgg.save_weights('saved_history/models/vggface.h5')

x,y,z = hyperparameters['height'], hyperparameters['width'], hyperparameters['channels']
X = Input((x,y,z))
model_list=[]
for task in range(hyperparameters['num_tasks']):
    # build model for the task
    tops = single_model(X, hyperparameters)  
    model = Model(inputs=X, outputs=[tops])
    model.load_weights('saved_history/models/vggface.h5', by_name=True)
    for i in model.layers:
        i._name+='_'+str(task)
    model_list.append(model)
    
final_model = Model(inputs = X, outputs = [model.output for model in model_list])
model,hyperparameters['is_trained']  = main(hyperparameters)
model.summary()

final_model.save_weights('saved_history/models/plain.h5')
model.load_weights('saved_history/models/plain.h5', by_name=True)


date = datetime.datetime.now().strftime("%d - %b - %y - %H:%M:%S")
filename = "model_joint_vgg" + date
logdir = "logs/" + filename

lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 1, mode = 'auto', min_lr = 0.00001)
filepath = "saved_history/models/model_joint" 
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='max', period = 1)
csv_logger = CSVLogger('saved_history/training_results_model_joint.csv', separator = ',', append=True)
tensorboard_callback = TensorBoard(log_dir = logdir)
def scheduler(epoch, lr):
    if epoch%3==0:
        return lr/10
lr_scheduler = LearningRateScheduler(scheduler)

preds = []
#model.load_weights("saved_history/models/vgg.h5")
#model.evaluate(val_gen)
#for i, d in enumerate([0, 1]):
#    with tf.device("/gpu:%d" %d):
#        with tf.name_scope("model_%d" %d) as scope:
history = model.fit_generator(train_gen,epochs=hyperparameters['epochs'], validation_data = val_gen,callbacks = [CalculatingPredictions(preds, filepath), lr, csv_logger,tensorboard_callback], verbose = 1)


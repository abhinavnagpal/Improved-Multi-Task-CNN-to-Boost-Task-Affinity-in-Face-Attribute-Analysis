import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dot,Dense, Softmax, Input, Flatten, Conv2D, BatchNormalization, MaxPooling2D, ReLU, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from matplotlib.patches import Rectangle
import keras

from hyperparameters import get_hyperparameters
from crossstitch import CrossStitch
from taskembeddings import taskEmbeddings

def addConvBlock_single(num_filters, kernel_size, hyperparameters, pool_size, tops, stride, pad, pool_stride, isPool , i, j, k = 0):
    
    input_tensor = tops
    conv = Conv2D(num_filters, kernel_size=kernel_size, name = 'conv'+str(i)+'_'+str(j),  strides=(stride, stride), padding = pad, kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l1(hyperparameters['reg_lambda']))(input_tensor)
    conv = ReLU()(conv)
    conv = BatchNormalization()(conv)
    if (isPool==True):
        if k==0:
            name = 'pool'+str(i)
        else:
            name = 'pool'+str(i)+str(k)
        conv = MaxPooling2D(pool_size=(pool_size, pool_size), name = name, strides = (pool_stride, pool_stride), padding='valid')(conv)
    tops= conv
  
    return tops

def single_model(X, hyperparameters):

# ------------------------------------------- Block 1 BEGINS ------------------------------------

    
    tops = addConvBlock_single(64, 3, hyperparameters, 3, X, 1, 'same', 2, False, 1 , 1)
    tops = addConvBlock_single(64, 3, hyperparameters, 3, tops, 1, 'same', 2, True, 1,2)

# ------------------------------------------- Block 1 ENDS ------------------------------------
            
# ------------------------------------------- Block 2 BEGINS ------------------------------------

    tops = addConvBlock_single(128, 3, hyperparameters, 3, tops, 1, 'same', 2, False, 2, 1)
    tops = addConvBlock_single(128, 3, hyperparameters, 3, tops, 1, 'same', 2, True,  2, 2)    

# ------------------------------------------- Block 2 ENDS ------------------------------------
# ------------------------------------------- Block 3 BEGINS ------------------------------------

    tops = addConvBlock_single(256, 3, hyperparameters, 3, tops, 1, 'same', 2, False, 3, 1)
    tops = addConvBlock_single(256, 3, hyperparameters, 3, tops, 1, 'same', 2, True,  3, 2, 1)
    tops = addConvBlock_single(256, 3, hyperparameters, 4, tops, 1, 'same', 2, True,  3 ,3, 2)
    
# ------------------------------------------- Block 3 ENDS ------------------------------------

# ------------------------------------------- Block 4 BEGINS ------------------------------------

    tops = addConvBlock_single(512, 3, hyperparameters, 3, tops, 1, 'same', 2, False, 4, 1)
    tops = addConvBlock_single(512, 3, hyperparameters, 3, tops, 1, 'same', 2, True, 4, 2, 1)
    tops = addConvBlock_single(512, 3, hyperparameters, 4, tops, 1, 'same', 2, True,  4, 3, 2)    
    
# ------------------------------------------- Block 4 ENDS ------------------------------------
    
    tops = Flatten(name = 'flat')(tops)
    tops = Dropout(hyperparameters["dropout_prob"])(tops) # task_embeddings[task_id]
#     tops = Dense(units = 64, name = 'dense', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops)
#     tops = BatchNormalization()(tops)
#     tops = ReLU()(tops)
    
    tops = Dense(units = 128, name='final_dense_1',activation = 'relu', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops)
    tops = BatchNormalization()(tops)
    tops = Dropout(hyperparameters["dropout_prob"])(tops)

    tops = Dense(units = 128, name='final_dense_2',activation = 'relu', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops)
    tops = BatchNormalization()(tops)
    tops = Dropout(hyperparameters["dropout_prob"])(tops)
    
    tops = Dense(units = 128, name='final_dense_3',activation = 'relu', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops)
    tops = BatchNormalization()(tops)

    tops = Dense(units = 1, name='output',activation = 'sigmoid', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(tops)
    return tops

def addConvBlock_joint(num_filters, kernel_size, hyperparameters, pool_size, tops, stride, pad, pool_stride, isPool , i, j, k = 0):
    for task_id in range(hyperparameters['num_tasks']):
        tops[task_id] = Conv2D(num_filters, kernel_size=kernel_size, name = 'conv'+str(i)+'_'+str(j)+'_'+str(task_id),  strides=(stride, stride), padding = pad, kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l1(hyperparameters['reg_lambda']))(tops[task_id])
        tops[task_id] = ReLU()(tops[task_id])
        tops[task_id] = BatchNormalization()(tops[task_id])
        
        if (isPool==True):
            if k==0:
                name = 'pool'+str(i)
            else:
                name = 'pool'+str(i)+str(k)
            
            tops[task_id] = MaxPooling2D(pool_size=(pool_size, pool_size), name = name+'_'+str(task_id), strides = (pool_stride, pool_stride), padding='valid')(tops[task_id])
  
    return tops

def architecture(hyperparameters):
    
    x,y,z = hyperparameters['height'], hyperparameters['width'], hyperparameters['channels']
    X = Input((x,y,z))
    
    sluice_outputs = {}
    for i in range(hyperparameters['num_tasks']):
        sluice_outputs['task_'+str(i)] = []
        
# ------------------------------------------- Block 1 BEGINS ------------------------------------


    tops = [X]*hyperparameters['num_tasks']
    tops = addConvBlock_joint(64, 3, hyperparameters, 3, tops, 1, 'same', 2, False, 1 , 1)
    tops = addConvBlock_joint(64, 3, hyperparameters, 3, tops, 1, 'same', 2, True, 1,2)
    
    ## CS
    if hyperparameters["enable_cs"]:
        cs1 = CrossStitch(hyperparameters['num_tasks'])(tops)
        tops = tf.unstack(cs1, axis=0)

            
# ------------------------------------------- Block 1 ENDS ------------------------------------
            
# ------------------------------------------- Block 2 BEGINS ------------------------------------

    tops = addConvBlock_joint(128, 3, hyperparameters, 3, tops, 1, 'same', 2, False, 2, 1)
    tops = addConvBlock_joint(128, 3, hyperparameters, 3, tops, 1, 'same', 2, True,  2, 2)    
    
    ## CS
    if hyperparameters["enable_cs"]:
        cs2 = CrossStitch(hyperparameters['num_tasks'])(tops)
        tops = tf.unstack(cs2, axis=0)

# ------------------------------------------- Block 2 ENDS ------------------------------------
# ------------------------------------------- Block 3 BEGINS ------------------------------------

    tops = addConvBlock_joint(256, 3, hyperparameters, 3, tops, 1, 'same', 2, False, 3, 1)
    tops = addConvBlock_joint(256, 3, hyperparameters, 3, tops, 1, 'same', 2, True, 3, 2, 1)
    tops = addConvBlock_joint(256, 3, hyperparameters, 4, tops, 1, 'same', 2, True,  3 ,3, 2)
    
    if hyperparameters["enable_cs"]:
        cs1 = CrossStitch(hyperparameters['num_tasks'])(tops)
        tops = tf.unstack(cs1, axis=0)
        
    if hyperparameters['enable_sluice']:
        for i in range(len(tops)):
            sluice_outputs['task_'+str(i)].append(Flatten()(tops[i]))
    
# ------------------------------------------- Block 3 ENDS ------------------------------------

# ------------------------------------------- Block 4 BEGINS ------------------------------------

    tops = addConvBlock_joint(512, 3, hyperparameters, 3, tops, 1, 'same', 2, False, 4, 1)
    tops = addConvBlock_joint(512, 3, hyperparameters, 3, tops, 1, 'same', 2, True, 4, 2, 1)
    tops = addConvBlock_joint(512, 3, hyperparameters, 4, tops, 1, 'same', 2, True,  4, 3, 2)    
    
    if hyperparameters["enable_cs"]:
        cs1 = CrossStitch(hyperparameters['num_tasks'])(tops)
        tops = tf.unstack(cs1, axis=0)
    
# ------------------------------------------- Block 4 ENDS ------------------------------------
    
    dynamic_weights = []
    task_embeddings = []
    loss_weights = []
    input_loss=[]
    
    for task_id in range(hyperparameters["num_tasks"]): 
        tops[task_id] = Flatten(name='flat' + '_'+ str(task_id))(tops[task_id])
        if hyperparameters['enable_sluice']:
            sluice_outputs['task_' + str(task_id)].append(tops[task_id])
            k = Concatenate()(sluice_outputs['task_' + str(task_id)])
            dynamic_weights.append(Dense(units = len(sluice_outputs['task_' + str(task_id)]), activation = 'softmax', name = 'weight_'+str(task_id))(k)) 
            task_embeddings.append(taskEmbeddings(hyperparameters['num_layers'])([sluice_outputs['task_' + str(task_id)],dynamic_weights[-1]]))                            
            task_embeddings[task_id] = Concatenate()(task_embeddings[task_id])
          
    for task_id in range(hyperparameters["num_tasks"]):
        task_embeddings[task_id] = Dense(units = 128, name='final_dense_x_1'+'_'+str(task_id), activation = 'relu', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(task_embeddings[task_id])
        task_embeddings[task_id] = BatchNormalization()(task_embeddings[task_id])

    for task_id in range(hyperparameters["num_tasks"]):
        task_embeddings[task_id] = Dropout(hyperparameters["dropout_prob"])(task_embeddings[task_id])
        
    for task_id in range(hyperparameters["num_tasks"]):
        task_embeddings[task_id] = Dense(units = 128, name='final_dense_2'+'_'+str(task_id), activation = 'relu', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(task_embeddings[task_id])
        task_embeddings[task_id] = BatchNormalization()(task_embeddings[task_id])

    for task_id in range(hyperparameters["num_tasks"]):
        task_embeddings[task_id] = Dropout(hyperparameters["dropout_prob"])(task_embeddings[task_id])

    for task_id in range(hyperparameters["num_tasks"]):
        task_embeddings[task_id] = Dense(units = 128, name='final_dense_3'+'_'+str(task_id), activation = 'relu', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(task_embeddings[task_id])
        task_embeddings[task_id] = BatchNormalization()(task_embeddings[task_id])

    for task_id in range(hyperparameters["num_tasks"]):
        task_embeddings[task_id] = Dense(units = 1, name='output'+'_'+str(task_id), activation = 'sigmoid', kernel_initializer=hyperparameters['initializer'], kernel_regularizer=regularizers.l2(hyperparameters['reg_lambda']))(task_embeddings[task_id])

    return X, task_embeddings


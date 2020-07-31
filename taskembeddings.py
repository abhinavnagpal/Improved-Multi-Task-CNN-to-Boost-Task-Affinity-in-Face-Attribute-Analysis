import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import datetime

from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Softmax, Input, Flatten, Conv2D, Activation, BatchNormalization, MaxPooling2D, ReLU, Dropout, Concatenate, Dot, Multiply, Softmax
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger, ReduceLROnPlateau, TensorBoard

from hyperparameters import get_hyperparameters

class taskEmbeddings(tf.keras.layers.Layer):
    def __init__(self, num_layers, **kwargs):
        self.num_layers = num_layers
        super(taskEmbeddings, self).__init__(**kwargs)

    def build(self, input_shape):
        super(taskEmbeddings, self).build(input_shape)

    def call(self, x):
        
        hyperparameters = get_hyperparameters()
        dynamic_weights = x[1]
        output_feature_maps = x[0]

        data = np.identity(len(output_feature_maps), dtype = np.float32)
        #onehotencoder = OneHotEncoder() # categorical_features = [0]
        #data = onehotencoder.fit_transform(np.arange(len(output_feature_maps)).reshape((-1,1))).toarray()
        weighted_output=[]
        for output in range(len(output_feature_maps)):
            one_hot = tf.constant(data[output])
            one_hot = tf.keras.backend.reshape(one_hot, shape=(-1, len(output_feature_maps)))
            one_hot = tf.repeat(one_hot, repeats=[hyperparameters['batch_size']], axis = 0)
            product1 = Dot(axes=1)([dynamic_weights, one_hot])
            product2 = Multiply()([product1, output_feature_maps[output]])
            weighted_output.append(product2)

        return weighted_output

    #def compute_output_shape(self, input_shape):
    #    return input_shape

    def get_config(self):
        base_config = super(taskEmbeddings, self).get_config()
        base_config['num_layers'] = self.num_layers
        return base_config

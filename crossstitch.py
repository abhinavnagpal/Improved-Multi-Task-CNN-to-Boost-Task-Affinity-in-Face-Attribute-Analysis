import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Softmax, Input, Flatten, Conv2D, BatchNormalization, MaxPooling2D, ReLU, Dropout, Concatenate, Dot, Multiply, Softmax
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback


class CrossStitch(tf.keras.layers.Layer):

    def __init__(self, num_tasks, *args, **kwargs):
        self.num_tasks = num_tasks
        super(CrossStitch, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='cs_kernel', shape=(self.num_tasks, self.num_tasks),initializer='identity',trainable=True)
        super(CrossStitch, self).build(input_shape)  

    def call(self, input_feature_maps):
        if (len(input_feature_maps)!=self.num_tasks):
            print("ERROR IN CROSS-STITCH")
      
        output_feature_maps = []
        for current_task in range(self.num_tasks):
            output = tf.math.scalar_mul(self.kernel[current_task,current_task], input_feature_maps[current_task])
            for other_task in range(self.num_tasks):
                if (current_task==other_task):
                    continue
          
                output+= tf.math.scalar_mul(self.kernel[current_task,other_task], input_feature_maps[other_task])
            output_feature_maps.append(output)
        return tf.stack(output_feature_maps, axis=0)
  
    def compute_output_shape(self, input_shape):
        return [self.num_tasks] + input_shape

    def get_config(self):
        base_config = super(CrossStitch, self).get_config()
        base_config['num_tasks'] = self.num_tasks
        return base_config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_generator(hyperparameters, train, val, test, attribute_id=None):
    
    # image_id = '000014.jpg'
    image_path = './dataset/celeba-dataset/'
    train_data_gen = ImageDataGenerator(rescale=1/255.0)
    test_data_gen = ImageDataGenerator(rescale=1/255.0)
    
    if attribute_id==None:
        targets = hyperparameters['targets'][1:]
        mode='multi_output'
    else:
        targets = hyperparameters['targets'][attribute_id+1]
        mode='raw'
    
    train_gen = train_data_gen.flow_from_dataframe(dataframe = train, 
                                     directory=image_path, 
                                     x_col = 'image_id', 
                                     y_col=targets, 
                                     class_mode = mode,
                                     target_size=(hyperparameters['height'], hyperparameters['width']), 
                                     batch_size = hyperparameters['batch_size']
                                     )

    val_gen = test_data_gen.flow_from_dataframe(dataframe = val, 
                                     directory=image_path, 
                                     x_col = 'image_id', 
                                     y_col=targets, 
                                     class_mode = mode,
                                     target_size=(hyperparameters['height'], hyperparameters['width']),
                                     batch_size = hyperparameters['batch_size']
                                     )

    test_gen = test_data_gen.flow_from_dataframe(dataframe = test, 
                                     directory=image_path, 
                                     x_col = 'image_id', 
                                     y_col=targets, 
                                     class_mode = mode,
                                     target_size=(hyperparameters['height'], hyperparameters['width']), 
                                     batch_size = hyperparameters['batch_size'])
    
    return train_gen, val_gen, test_gen

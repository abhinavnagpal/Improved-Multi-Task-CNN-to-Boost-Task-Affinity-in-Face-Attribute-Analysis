import numpy as np
import pandas as pd

def preprocess(hyperparametersattr, eval_partition, attribute_id = None):
    
    if attribute_id==None:
        attr = attr[hyperparameters['targets']]
        print(1)
    else:
        attr = attr[['image_id', hyperparameters['targets'][attribute_id+1]]]

    attr = attr[hyperparameters['targets']]
    attr = attr.replace(-1, 0)
    attr = attr.set_index('image_id')
    eval_partition = eval_partition.set_index('image_id')
    attr = attr.join(eval_partition)
    attr['image_id'] = attr.index
    
    train = attr.loc[attr['partition']==0]
    val = attr.loc[attr['partition']==1]
    test = attr.loc[attr['partition']==2]
    
    train = train.drop('partition', axis=1)
    val = val.drop('partition', axis=1)
    test = test.drop('partition', axis=1)

    return (train, val, test)

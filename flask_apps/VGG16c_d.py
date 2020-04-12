# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:40:44 2020

@author: Ali
"""
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np

def initialize_sess():
    global sess
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
        )
    return tf.Session(config=config)

def get_model(path):
    model = load_model(path)
    model._make_predict_function()
    graph = tf.get_default_graph()
    return model, graph


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

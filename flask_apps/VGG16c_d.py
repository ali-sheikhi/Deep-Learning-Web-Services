# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:40:44 2020

@author: Ali
"""
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse
import os



def get_model(path):
    model = load_model(path)
    model._make_predict_function()
    #graph = tf.get_default_graph()
    #return model, graph
    return model


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help="Path of image to detect objects", default="Test images - classification/cat.jpg")
    args = parser.parse_args()
    image_path = args.image_path
    wpath= os.path.join("VGG16","VGG16_cats_and_dogs.h5")
    model_vgg, graph_vgg = get_model(wpath)
    with graph_vgg.as_default():
        image = Image.open(image_path)
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = model_vgg.predict(processed_image).tolist()
    print("Prediction for Dog: ",prediction[0][0])
    print("Prediction for Cat: ",prediction[0][1])
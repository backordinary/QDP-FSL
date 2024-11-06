# https://github.com/shourya2002-geek/Quantum-Ecosystem-for-Efficient-Medical-Imaging/blob/14b5d9a79f2176c5013bbea1327f842166f2a4ac/model.py
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Model
import os
from keras.applications.resnet import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import h5py

import qiskit
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl  

def filename_to_label(ch):
    switcher = {
        'M': 0,
        'H': 1,
        'P': 2,
        'N': 3
    }
    return switcher.get(ch)


def crop(img):
    x_start = 140
    x_end = -40
    y_start = 287
    y_end = -57

    img = img[y_start: y_end, x_start: x_end]
    return img


def remove_bg(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 50:
                img[i][j] = 0
            else:
                img[i][j] = 255


def remove_verticals(img):
    x1_start = 503
    x1_end = 506
    y1_start = 125
    y1_end = 825

    img[y1_start: y1_end, x1_start: x1_end] = 255

    x2_start = 995
    x2_end = 998
    y2_start = 125
    y2_end = 825

    img[y2_start: y2_end, x2_start: x2_end] = 255

    x3_start = 1486
    x3_end = 1490
    y3_start = 125
    y3_end = 825

    img[y3_start: y3_end, x3_start: x3_end] = 255


def build_model():

    pickle_in1 = open('finalized_model.sav', 'rb')  
    model = pkl.load(pickle_in1)  
   

    prediction_folder = './AllNew'
    filename = "HB(47).jpg"
    pred_img = cv2.imread(os.path.join(
        prediction_folder, filename), cv2.IMREAD_GRAYSCALE)
    pred_img = crop(pred_img)
    remove_bg(pred_img)
    remove_verticals(pred_img)

    base_model = load_model('./resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    resnet_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


    folder = "./ECG_postprocess"
    image_size = 224
    filename = "HB(47).jpg"
    test_features_array = np.empty([0, 2048])
    prediction_image = tf.keras.preprocessing.image.load_img(
        os.path.join(folder, filename), target_size=(image_size, image_size))
    x = tf.keras.preprocessing.image.img_to_array(prediction_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print(x.shape)

    test_features = resnet_model.predict(x)
    test_features = test_features.reshape(2048,)
    test_features_array = np.vstack([test_features_array, test_features])

    test_X_SVD = tsvd.transform(test_features_array)

    test_X = scaler.transform(test_X_SVD)

    predicts = model.predict(test_X)
    st.write("prediction:", predicts)


# https://github.com/MarkCodering/NTU-Quantum-Research-QCNN-Demo/blob/04003b6543d11f667f380fc718a8ff7ea1072f5a/singleQubitEncoding.py
# Get MNIST Dataset
# Encode the image into a single qubit
# Run the circuit
# Measure the result
# Compare the result with the label
# Calculate the accuracy
# Plot the result

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit.visualization import plot_state_qsphere
#import tensorflow as tf

SAVE_PATH = "./simple_mnist/"
x_encodedImage = []

def TestSingleQubitEncoding():
    # Get MNIST Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Print the information about the single image of the dataset with height and width
    print("x_train.shape: ", x_train.shape)
    x_encodedImage = np.zeros((len(x_train), 1, 28, 28))
    x_train[0] = x_train[0].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[0][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[0][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[0])
    x_train[1] = x_train[1].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[1][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[1][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[1])
    x_train[2] = x_train[2].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[2][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[2][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[2])
    x_train[3] = x_train[3].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[3][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[3][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[3])
    x_train[4] = x_train[4].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[4][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[4][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[4])
    x_train[5] = x_train[5].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[5][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[5][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[5])
    x_train[6] = x_train[6].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[6][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[6][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[6])
    x_train[7] = x_train[7].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[7][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[7][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[7])
    x_train[8] = x_train[8].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[8][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[8][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[8])
    x_train[9] = x_train[9].reshape(1, 28, 28)
    for j in range(0, 28):
        for k in range(0, 28):
            # Collect the pixel value
            # r = (drow + frow); c = (dcol + fcol)
            # drow = 0; frow = 0; dcol = 0; fcol = 0
            # r = 0; c = 0
            r = j
            c = k
            pixel = x_train[9][r][c]
            
            if(r<28 and c<28):
                #x append value at pixel (r, c) ∈ d
                x_encodedImage[9][0][r][c] = pixel
            else:
                pixel.append(0)

    print("x_encodedImage: ", x_encodedImage[9])

#TestSingleQubitEncoding()
#Use the qiskit to plot the first quantum image
# Convert the first quantum image to the density matrix
# Plot the density matrix
qc = QuantumCircuit(2)
qc.h([0, 1])
qc.cz(0,1)
qc.ry(np.pi/3, 0)
qc.rx(np.pi/5, 1)
qc.z(1)

matrix = DensityMatrix(qc)
plot_state_qsphere(matrix,
     show_state_phases = True, use_degrees = True)
# https://github.com/wei-coding/NCHU-quantum-information/blob/9101228a8c82464d51012cc56337a7f12ad271cc/final_report/detector.py
from qiskit import *
from utils.helper import amplitude_encode

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from PIL import Image
style.use('bmh')

import argparse

image_crop_size = 32   # Width of each part of image for processing
data_qb = 10 # log2(8 * 8)
anc_qb = 1
total_qb = data_qb + anc_qb

def detector(filename, outputname):
    image_raw = np.array(Image.open(filename))
    print('Raw Image info:', image_raw.shape)
    print('Raw Image datatype:', image_raw.dtype)

    image_size = image_raw.shape[0]

    image = []
    for i in range(image_size):
        image.append([])
        for j in range(image_size):
            image[i].append(image_raw[i][j][0] / 255)
            
    image = np.array(image)
    print('Image shape (numpy array):', image.shape)

    parts = int(image_size / image_crop_size)
    cropped_img = [[0 for _ in range(parts)] for _ in range(parts)]
    for i in range(parts):
        for j in range(parts):
            cropped_img[i][j] = image[i*image_crop_size:(i+1)*image_crop_size, j*image_crop_size:(j+1)*image_crop_size]

    D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)

    result = [[0 for _ in range(parts)] for _ in range(parts)]
    back = Aer.get_backend('statevector_simulator')

    for i in range(parts):
        for j in range(parts):
            print(f"part ({i}, {j})")

            image_norm_h = amplitude_encode(cropped_img[i][j])
            image_norm_v = amplitude_encode(cropped_img[i][j].T)

            # 水平的
            qc_h = QuantumCircuit(total_qb)
            qc_h.initialize(image_norm_h, range(1, total_qb))
            qc_h.h(0)
            qc_h.unitary(D2n_1, range(total_qb))
            qc_h.h(0)

            # 垂直的
            qc_v = QuantumCircuit(total_qb)
            qc_v.initialize(image_norm_v, range(1, total_qb))
            qc_v.h(0)
            qc_v.unitary(D2n_1, range(total_qb))
            qc_v.h(0)

            circ_list = [qc_h, qc_v]
                    
            
            results = execute(circ_list, backend=back).result()
            sv_h = results.get_statevector(qc_h)
            sv_v = results.get_statevector(qc_v)

            edge_h = np.abs(np.array([sv_h[2*i+1].real for i in range(2**data_qb)])).reshape(image_crop_size, image_crop_size)
            edge_v = np.abs(np.array([sv_v[2*i+1].real for i in range(2**data_qb)])).reshape(image_crop_size, image_crop_size).T

            edge_h[0, :] = 0
            edge_h[edge_h.shape[0]-1, :] = 0
            edge_h[:, 0] = 0
            edge_h[:, edge_h.shape[0]-1] = 0
            edge_v[0, :] = 0
            edge_v[edge_v.shape[0]-1, :] = 0
            edge_v[:, 0] = 0
            edge_v[:, edge_v.shape[0]-1] = 0

            edge = edge_h + edge_v

            result[i][j] = edge

    result_np = np.ndarray((image_size, image_size))
    for i in range(parts):
        for j in range(parts):
            result_np[i*image_crop_size:(i+1)*image_crop_size, j*image_crop_size:(j+1)*image_crop_size] = result[i][j]

    plt.imsave(outputname, result_np)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", "-i", type=str, required=True)
    parser.add_argument("--outfile", "-o", type=str)

    args = parser.parse_args()

    if args.outfile:
        detector(args.infile, args.outfile)
    else:
        detector(args.infile, "output.png")

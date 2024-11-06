# https://github.com/paritoshkc/Quantum-Computing/blob/410bc4861e64dc92753af0080469dc9e1588f911/runner_320.py

# frqi circuit from https://github.com/Shedka/citiesatnight


import utils
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere
from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout
from qiskit import execute, Aer, BasicAer
from qiskit.providers.aer.noise import NoiseModel
import numpy as np
import matplotlib.pyplot as plt
from resizeimage import resizeimage
from PIL import Image, ImageOps
import frqi
# import quantum_edge_detection as qed



# Insert API key generated after registring in IBM Quantum Experience
# IBMQ.save_account('API KEY')

IBMQ.load_account()
provider = IBMQ.get_provider( group='open', project='main')


# dimensions of the image
size=32

#target image
images=utils.get_Cat_320_image()

# New blank image
new_im = Image.new('RGB', (320, 320))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*0),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*0,32*9))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*1),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*1,32*9))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*2),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*2,32*9))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*3),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*3,32*9))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*4),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*4,32*9))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*5),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*5,32*9))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*6),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*6,32*9))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*7),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*7,32*9))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*8),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*8,32*9))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*0))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[0] for 0 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*0))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*1))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[1] for 1 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*1))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*2))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[2] for 2 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*2))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*3))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[3] for 3 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*3))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*4))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[4] for 4 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*4))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*5))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[5] for 5 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*5))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*6))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[6] for 6 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*6))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*7))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[7] for 7 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*7))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*8))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[8] for 8 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*8))
normalized_image=utils.large_image_normalization(images,32+(32*9),32+(32*9))
genimg= np.array([])
anc = QuantumRegister(1, "anc")
img = QuantumRegister(11, "img")
anc2 = QuantumRegister(1, "anc2")
c = ClassicalRegister(12)
qc = QuantumCircuit(anc, img, anc2, c)

for i in range(1, len(img)):
        qc.h(img[i])


for i in range(len(normalized_image)):
        if normalized_image[i] != 0:
                frqi.c10mary(qc, 2 * normalized_image[i], format(i, '010b'), img[0], anc2[0], [img[9] for 9 in range(1,len(img))])
qc.measure(img, c[1:12])
print(qc.depth())
numOfShots = 1000000
result = execute(qc, Aer.get_backend('qasm_simulator'), shots=numOfShots).result()
for i in range(len(normalized_image)):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])

genimg *= 32.0 * 255.0 
genimg = genimg.astype('int')
genimg = genimg.reshape((32,32))
im=Image.fromarray(genimg)

new_im.paste(im,(32*9,32*9))
new_im.show()
new_im.save('Result_320.png')


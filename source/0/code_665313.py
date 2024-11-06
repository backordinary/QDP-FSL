# https://github.com/mahsakarimii/The-power-of-one-clean-qubit-in-supervised-machine-learning./blob/9870f39dc40d457aa05fa1b5ac4062bb29067348/code.py
#import what we want
from qiskit import *

import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn import svm, datasets

from qiskit import BasicAer

# Tomography functions
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import qiskit.ignis.mitigation.measurement as mc

# example dataset
from qiskit import QuantumCircuit, BasicAer, transpile

from sklearn.model_selection import train_test_split

from qiskit_machine_learning.datasets import ad_hoc_data


#adhoc dataset
adhoc_dimension = 2
train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=20,
    test_size=5,
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

import time


feature1=train_features[:,0]
feature2=train_features[:,1]
feature3=test_features[:,0]
feature4=test_features[:,1]

#circuit
from qiskit.circuit import Parameter
rho=[]
for i in range(len(train_features)):
  for j in range(len(train_features)):
      qc = QuantumCircuit(5)
      #circuit preparation
      qc.h(0)
      qc.h(1)
      qc.cx(1,3)
      qc.h(2)
      qc.cx(2,4)
      #l=2
      qc.ch(0,1)
      qc.ch(0,2)
      qc.crz(2*feature1[i],[0],[1])
      qc.crz(2*feature2[i],[0],[2])
      qc.ccx(0,1,2)
      qc.crz(2*(np.pi-feature1[i])*(np.pi-feature2[i]),[0],[2])
      qc.ccx(0,1,2)
      #
      qc.ch(0,1)
      qc.ch(0,2)
      qc.crz(2*feature1[i],[0],[1])
      qc.crz(2*feature2[i],[0],[2])
      qc.ccx(0,1,2)
      qc.crz(2*(np.pi-feature1[i])*(np.pi-feature2[i]),[0],[2])
      qc.ccx(0,1,2)
      #
      qc.ccx(0,1,2)
      qc.crz(-2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
      qc.ccx(0,1,2)
      qc.crz(-2*feature2[j],[0],[2])
      qc.crz(-2*feature1[j],[0],[1])
      qc.ch(0,2)
      qc.ch(0,1)
      #
      qc.ccx(0,1,2)
      qc.crz(-2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
      qc.ccx(0,1,2)
      qc.crz(-2*feature2[j],[0],[2])
      qc.crz(-2*feature1[j],[0],[1])
      qc.ch(0,2)
      qc.ch(0,1)
      #
      t=time.time()
      qst_circuit = state_tomography_circuits(qc,[0])
      job = qiskit.execute(qst_circuit, Aer.get_backend('qasm_simulator'), shots=8000)
      tomo_circuit = StateTomographyFitter(job.result(), qst_circuit)
      rho_circuit = tomo_circuit.fit()
      rho.append(rho_circuit)
print('Time taken:', time.time() - t)  
print(rho)
print(qc)

#just picking the element \rho_{11} of all the density matrices.
ph=np.zeros(len(rho),dtype=np.complex_)
for i in range(len(rho)):
  ph[i]=rho[i][0][1]

#element \rho_{11}s are complex numbers with abs command we calculate their absolute value. 
# Here, coefficient two reffers to the 1/2 coefficient outside of the density matrix.
pp=[]
for i in ph:
  pp.append(2*(np.abs(i)))

#Let's put them into a 40*40 matrix (Gram matrix)
k = np.reshape(pp,(40,40))
#Let's print the diagonal elements to see wether they are equal two one or not.
d=[]
for i in range(40):
  d.append(k[i][i])
# Great, so they are really one. This is Python accuracy!
d


# The same circuit, this time for testing datasets
from qiskit.circuit import Parameter
rho_test=[]
for i in range(len(test_features)):
  for j in range(len(train_features)):
      qc = QuantumCircuit(5)
      #input state preparation
      qc.h(0)
      qc.h(1)
      qc.cx(1,3)
      qc.h(2)
      qc.cx(2,4)
      #L=2
      qc.ch(0,1)
      qc.ch(0,2)
      qc.crz(2*feature3[i],[0],[1])
      qc.crz(2*feature4[i],[0],[2])
      qc.ccx(0,1,2)
      qc.crz(2*(np.pi-feature3[i])*(np.pi-feature4[i]),[0],[2])
      qc.ccx(0,1,2)
      #
      qc.ch(0,1)
      qc.ch(0,2)
      qc.crz(2*feature3[i],[0],[1])
      qc.crz(2*feature4[i],[0],[2])
      qc.ccx(0,1,2)
      qc.crz(2*(np.pi-feature3[i])*(np.pi-feature4[i]),[0],[2])
      qc.ccx(0,1,2)
      #
      qc.ccx(0,1,2)
      qc.crz(-2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
      qc.ccx(0,1,2)
      qc.crz(-2*feature2[j],[0],[2])
      qc.crz(-2*feature1[j],[0],[1])
      qc.ch(0,2)
      qc.ch(0,1)
      #
      qc.ccx(0,1,2)
      qc.crz(-2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
      qc.ccx(0,1,2)
      qc.crz(-2*feature2[j],[0],[2])
      qc.crz(-2*feature1[j],[0],[1])
      qc.ch(0,2)
      qc.ch(0,1)
      #
      t=time.time()
      qst_circuit = state_tomography_circuits(qc,[0])
      job = qiskit.execute(qst_circuit, Aer.get_backend('qasm_simulator'), shots=8000)
      tomo_circuit = StateTomographyFitter(job.result(), qst_circuit)
      rho_circuit = tomo_circuit.fit()
      rho_test.append(rho_circuit)
print('Time taken:', time.time() - t)  
print(rho_test)
print(qc)

#just picking the element \rho_{11} of all the density matrices.
pht=np.zeros(len(rho_test),dtype=np.complex_)
for i in range(len(rho_test)):
  pht[i]=rho_test[i][0][1]

#element \rho_{11}s are complex numbers with abs command we calculate their absolute value. 
# Here, coefficient two reffers to the 1/2 coefficient outside of the density matrix.
ppt=[]
for j in pht:
  ppt.append(2*(np.abs(j)))

#Let's put them into a 10*40 matrix (Gram matrix for testing datasets)
kt = np.reshape(ppt,(10,40))

from sklearn.svm import SVC

#Train the model
clf = svm.SVC(kernel='precomputed')
gram_train = k
Y_train=clf.fit(gram_train, train_labels)

#Test the model
gram_test = kt
Y_predict=clf.predict(gram_test)

#Let's print to see what our model predicted.
Y_predict

#Let's just print the test labels to do a comparrision with what we have as prediction.
test_labels

# Let's calculte the accuracy.
accuracy_score(Y_predict, test_labels)

# Let's calculate the accuracy with another method, just to make sure.
clf.score(kt,test_labels)

#Let's draw our teaining and predicting datapoints
# This cell also can be found in Qiskit webpage.
plt.figure(figsize=(5, 5))
plt.ylim(0, 2 * np.pi)
plt.xlim(0, 2 * np.pi)
plt.imshow(
    np.asmatrix(adhoc_total).T,
    interpolation="nearest",
    origin="lower",
    cmap="RdBu",
    extent=[0, 2 * np.pi, 0, 2 * np.pi],
)

plt.scatter(
    train_features[np.where(train_labels[:] == 0), 0],
    train_features[np.where(train_labels[:] == 0), 1],
    marker="s",
    facecolors="w",
    edgecolors="b",
    label="A train",
)
plt.scatter(
    train_features[np.where(train_labels[:] == 1), 0],
    train_features[np.where(train_labels[:] == 1), 1],
    marker="o",
    facecolors="w",
    edgecolors="r",
    label="B train",
)
plt.scatter(
    test_features[np.where(Y_predict[:] == 0), 0],
    test_features[np.where(Y_predict[:] == 0), 1],
    marker="s",
    facecolors="b",
    edgecolors="w",
    label="A test",
)
plt.scatter(
    test_features[np.where(Y_predict[:] == 1), 0],
    test_features[np.where(Y_predict[:] == 1), 1],
    marker="o",
    facecolors="r",
    edgecolors="w",
    label="B test",
)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
plt.title("Ad hoc dataset")

plt.show()

#Let's visualize kernel matrix (it is called k in this notebook.)
plt.figure(figsize=(5, 5))
plt.imshow(np.asmatrix(k),interpolation='nearest', origin='upper')
plt.title("Analytical Kernel Matrix")
plt.colorbar()
plt.show()

#Some mathematical formula to calculate coherence
x1=np.abs(1-k)/2

h1=-x1*np.log2(x1)-(1-x1)*np.log2(1-x1)

np.max(h1)

#Let's put this 1600 values for coherence into a 40*40 matrix to visualize it as a square.
h2=np.reshape(h1,(40,40))

plt.figure(figsize=(5, 5))
plt.ylim(0, 2 * np.pi)
plt.xlim(0, 2 * np.pi)
plt.imshow(
    np.asmatrix(h2),
    interpolation="nearest",
    origin="upper",
    extent=[0, 2 * np.pi, 0, 2 * np.pi], vmin=0 , vmax=1
)
plt.colorbar()
plt.show()

#mathematical formula to calculate discord
dis=1/16*(1-k)

np.max(dis)

#Let's visulaize discord
plt.figure(figsize=(5, 5))
plt.ylim(0, 2 * np.pi)
plt.xlim(0, 2 * np.pi)
plt.imshow(
    np.asmatrix(dis),
    interpolation="nearest",
    origin="upper",
    extent=[0, 2 * np.pi, 0, 2 * np.pi], vmin=0 , vmax=0.063
)
plt.colorbar()
plt.show()

#That's it!
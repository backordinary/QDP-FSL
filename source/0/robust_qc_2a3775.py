# https://github.com/yuxuan-du/Robust-quantum-classifier/blob/d5530006ab1314914e67a689f7a06cacf211bb2a/robust-QC.py
import pennylane as qml
import pyquil
from pennylane import numpy as np
import qiskit
import qiskit.providers.aer.noise as noise
import time
import os
# load synthetic dataset based on the paper 'Supervised learning with quantum-enhanced feature spaces'
data_all = np.load('data.npy')
label_all = np.load('label.npy')

data_train, label_train = data_all[:100], label_all[:100]
data_vali, label_vali = data_all[100:200], label_all[100:200]
data_test, label_test = data_all[200:300], label_all[200:300]

# number of qubits in the circuit
nr_qubits = 3
# number of layers in the circuit
nr_layers = 3
encode_layers = 1
size = 100
params = np.random.uniform(0, np.pi * 2, (nr_layers,   nr_qubits))  
params_opt = np.zeros((nr_layers,   nr_qubits))
vali_acc_base = 0

def train_result_record():
    return {
            'loss': [],
            'train_acc': [],
            'valid_acc': [],
            'test_acc': []
            }

records = train_result_record()

# encoder
def encode_layer(feature, j):
 
    for i in range(nr_qubits):
        qml.RY( feature[i], wires=i)
    
    phi = (np.pi - feature[0].val)*(np.pi - feature[1].val)*(np.pi - feature[2].val)


def layer(params, j):
    for i in range(nr_qubits):
        qml.RY(params[j,   i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])


prob_1 = 0.05   
prob_2 = 0.2  
error_1 = noise.depolarizing_error(prob_1, 1)
error_2 = noise.depolarizing_error(prob_2, 2)
noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
dev = qml.device('qiskit.aer', wires=3, noise_model=noise_model)

@qml.qnode(dev)
def circuit(feature, params, A=None):
    for j in range(encode_layers):
        encode_layer(feature, j)
    for j in range(nr_layers):
        layer(params, j)
    return qml.expval(qml.Hermitian(A, wires=[0, 1, 2]))


opt = qml.AdamOptimizer(0.05)
def cost_fn(params):
    global data_train, label_train
    loss = 0
    indices = np.arange(data_train.shape[0])  
    data_train = data_train[indices]
    label_train = label_train[indices]
    correct = 0
    for data, label in zip(data_train[:size], label_train[:size]):
        out = circuit(data, params, A=np.kron(np.eye(4), np.array([[1, 0], [0, 0]])))
        loss +=   (label - out)**2
        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    loss /= size
    print('loss: {} , acc: {} '.format(loss, correct / size))
    records['train_acc'].append(correct / size)
    records['loss'].append(loss._value)
    return loss

def test_fn(params):
    correct = 0
    for data, label in zip(data_test, label_test):
        out = circuit(data, params, A=np.kron(np.eye(4), np.array([[1, 0], [0, 0]])))

        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    print('Test acc: {}'.format(correct / label_test.shape[0]))
    records['test_acc'].append(correct / label_test.shape[0])

def valid_fn(params):
    correct = 0
    for data, label in zip(data_vali, label_vali):
        out = circuit(data, params, A=np.kron(np.eye(4), np.array([[1, 0], [0, 0]])))

        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    print('Valid acc: {}'.format(correct / label_vali.shape[0]))
    records['valid_acc'].append(correct / label_vali.shape[0])
    return correct / label_vali.shape[0]

for i in range(400):
    print('Epoch {}'.format(i))
    params = opt.step(cost_fn, params)
    # if (i+1) % 10 == 0:
    valid_acc = valid_fn(params)
    if valid_acc > vali_acc_base:
        params_opt = params
    f = open('train_result' + 'robust'+ '.txt', 'w')
    f.write(str(records))
    f.close()

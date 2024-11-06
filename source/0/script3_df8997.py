# https://github.com/claudioalvesmonteiro/QuantumNeuronRealWeights/blob/ab295615a035d2d3b1cf574c64c15cc7c2c89d3f/script3.py
from encodingv2 import *
from sf import sfGenerator
from hsgs import hsgsGenerator
import numpy as np
import math
import random
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit, ClassicalRegister
from sympy.combinatorics.graycode import GrayCode
from qiskit.aqua.utils.controlled_circuit import apply_cu3
from extrafunctions import *
from neuron import *

np.random.seed(7)

# df = pd.read_csv('dataset.csv')
# listOfInput = df.iloc[:, :-1].to_numpy()
# listOfExpectedOutput = df.iloc[:, -1].to_numpy()

# data_len = len(listOfExpectedOutput)
input_dim = 2
# nb_epochs = 20
lr = 0.001
threshold = none
simulator = Aer.get_backend('qasm_simulator')

# print("Comparativo de execução do produto interno")

inputArr = []
weightsArr = []
diffArr = []
sizeArr = []
input_dim = 2 ** 2
print("Level", 2, "- Input size:", input_dim)

for i in range(5000):
    w = np.random.uniform(-1, 1, input_dim) #np.random.rand(input_dim) # Real weights
    w = normalize(w)
    inputVector = deterministicBinarization(np.random.uniform(-1, 1, input_dim))

    encodingcircuit = createNeuron(inputVector, w, "encoding-weight")     
    encodingResult = executeNeuron(encodingcircuit, simulator, threshold=None)


    classicalResult = np.sum(np.multiply(inputVector, w)) ** 2

    inputArr.append(inputVector)
    weightsArr.append(w)
    diffArr.append(round(np.sqrt((classicalResult-encodingResult) ** 2), 3))
    sizeArr.append(input_dim)

    # print("#", i)
input_dim = 2 ** 3
print("Level", 3, "- Input size:", input_dim)

for i in range(5000):
    w = np.random.uniform(-1, 1, input_dim) #np.random.rand(input_dim) # Real weights
    w = normalize(w)
    inputVector = deterministicBinarization(np.random.uniform(-1, 1, input_dim))

    encodingcircuit = createNeuron(inputVector, w, "encoding-weight")     
    encodingResult = executeNeuron(encodingcircuit, simulator, threshold=None)


    classicalResult = np.sum(np.multiply(inputVector, w)) ** 2

    inputArr.append(inputVector)
    weightsArr.append(w)
    diffArr.append(round(np.sqrt((classicalResult-encodingResult) ** 2), 3))
    sizeArr.append(input_dim)

    # print("#", i)
input_dim = 2 ** 4
print("Level", 4, "- Input size:", input_dim)

for i in range(5000):
    w = np.random.uniform(-1, 1, input_dim) #np.random.rand(input_dim) # Real weights
    w = normalize(w)
    inputVector = deterministicBinarization(np.random.uniform(-1, 1, input_dim))

    encodingcircuit = createNeuron(inputVector, w, "encoding-weight")     
    encodingResult = executeNeuron(encodingcircuit, simulator, threshold=None)


    classicalResult = np.sum(np.multiply(inputVector, w)) ** 2

    inputArr.append(inputVector)
    weightsArr.append(w)
    diffArr.append(round(np.sqrt((classicalResult-encodingResult) ** 2), 3))
    sizeArr.append(input_dim)

    # print("#", i)
input_dim = 2 ** 5
print("Level", 5, "- Input size:", input_dim)

for i in range(5000):
    w = np.random.uniform(-1, 1, input_dim) #np.random.rand(input_dim) # Real weights
    w = normalize(w)
    inputVector = deterministicBinarization(np.random.uniform(-1, 1, input_dim))

    encodingcircuit = createNeuron(inputVector, w, "encoding-weight")     
    encodingResult = executeNeuron(encodingcircuit, simulator, threshold=None)


    classicalResult = np.sum(np.multiply(inputVector, w)) ** 2

    inputArr.append(inputVector)
    weightsArr.append(w)
    diffArr.append(round(np.sqrt((classicalResult-encodingResult) ** 2), 3))
    sizeArr.append(input_dim)

    # print("#", i)
input_dim = 2 ** 6
print("Level", 6, "- Input size:", input_dim)

for i in range(5000):
    w = np.random.uniform(-1, 1, input_dim) #np.random.rand(input_dim) # Real weights
    w = normalize(w)
    inputVector = deterministicBinarization(np.random.uniform(-1, 1, input_dim))

    encodingcircuit = createNeuron(inputVector, w, "encoding-weight")     
    encodingResult = executeNeuron(encodingcircuit, simulator, threshold=None)


    classicalResult = np.sum(np.multiply(inputVector, w)) ** 2

    inputArr.append(inputVector)
    weightsArr.append(w)
    diffArr.append(round(np.sqrt((classicalResult-encodingResult) ** 2), 3))
    sizeArr.append(input_dim)

    # print("#", i)

def sortWithRespect(inputV, weights, sizeInput, diff):
    '''
    Sorts a with respect to b and sorts b too
    In the end the indexes of a and b will still be corresponded to each other
    '''
    weights = np.asarray(weights)
    inputV = np.asarray(inputV)
    sizeInput = np.asarray(sizeInput)
    diff = np.asarray(diff)
    idx_sorted = diff.argsort()
    
    return (diff[idx_sorted][::-1], sizeInput[idx_sorted][::-1], inputV[idx_sorted][::-1], weights[idx_sorted][::-1])


(a, b, c, d) = sortWithRespect(inputArr, weightsArr, sizeArr, diffArr)


print("diff", ',', "input_dim", ',',"input", ',', "weights")        
for i in range(len(inputArr)):
    print(a[i], ',', b[i],',', c[i],',', d[i].tolist())
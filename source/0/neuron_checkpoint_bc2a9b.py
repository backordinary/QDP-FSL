# https://github.com/claudioalvesmonteiro/quantumML-health/blob/e793af307d3a3d95787b1b17b97641f158b3e936/code/.ipynb_checkpoints/neuron-checkpoint.py
from code.encodingsource import InitializerUniformlyRotation
from code.sf import sfGenerator
from code.hsgs import hsgsGenerator
from code.phaseEncoding2 import phaseEncodingGenerator
from code.classical_pso import PSO
import numpy as np
import math
import random
from qiskit import execute, Aer, QuantumRegister, QuantumCircuit, ClassicalRegister
from sympy.combinatorics.graycode import GrayCode
# from qiskit.aqua.circuits.gates.multi_control_rotation_gates  import _apply_mcu3_graycode, mcrx
#from qiskit.aqua.utils.controlled_circuit import apply_cu3
from code.extrafunctions import *

from code.encodingv2 import *

np.random.seed(7)

def inverterSinal(v):
    for i in range(len(v)):
        v[i] *=-1

def encodingGenerator(inputVector, circuit, q_input):
	#inputVector is a Python list 
		#eg. inputVector=[1, -1, 1, 1]
	#circuit is a Quantum Circuit
	#q_input is the input qubits
	QuantumCircuit.ur_initialize = InitializerUniformlyRotation.initialize
	circuit.ur_initialize(inputVector, q_input)

	return circuit


def encodingGenerator2(inputVector, circuit, q_input):
	#inputVector is a Python list 
		#eg. inputVector=[1, -1, 1, 1]
	#circuit is a Quantum Circuit
	#q_input is the input qubits
    
	if (inputVector[0] == -1):
		inverterSinal(inputVector)
	Encoding(circuit, inputVector, q_input).qcircuit
	n = int(math.log(len(inputVector),2))
	for i in range(n//2):
		circuit.swap(q_input[i], q_input[n-i-1])
	#for i in range(n):
	#	circuit.x(q_input[i])
	return circuit


def createNeuron (inputVector, weightVector, circuitGeneratorOfUOperator, ancilla=False):  
	n = int(math.log(len(inputVector), 2))

	circuit = QuantumCircuit()
	q_input = QuantumRegister(n, 'q_input')
	#q_target = QuantumRegister(1, 'q_target')
	q_output = QuantumRegister(1, 'q_output')
	c_output = ClassicalRegister(1, 'c_output')
	circuit.add_register(q_input)
	#circuit.add_register(q_target)
	circuit.add_register(q_output)
	circuit.add_register(c_output)
           
	if ancilla == True:
		q_aux = QuantumRegister(n-1, 'q_aux')
		circuit.add_register(q_aux)
	else:
		q_aux = None

                                        
	if circuitGeneratorOfUOperator == "phase-encoding":
			for i in range(n):
				circuit.h(q_input[i])
                
			inputVector = [i*math.pi for i in inputVector]
			weightVector = [-i*math.pi for i in weightVector]
            
			phaseEncodingGenerator(inputVector, circuit, q_input, n)
			phaseEncodingGenerator(weightVector, circuit, q_input, n, weight=True)          
                
                
	elif circuitGeneratorOfUOperator == "hsgs":
			for i in range(n):
				circuit.h(q_input[i])
                
			#inputVectorbin = deterministicBinarization(inputVector)
			#weightVectorbin = deterministicBinarization(weightVector)
			#inputVectorbin = [i*math.pi for i in inputVectorbin]
			#weightVectorbin = [-i*math.pi for i in weightVectorbin]
            
			hsgsGenerator(inputVector, circuit, q_input, n)
			hsgsGenerator(weightVector, circuit, q_input, n)

            
	elif circuitGeneratorOfUOperator == "sf":
		for i in range(n):
			circuit.h(q_input[i])
		sfGenerator(inputVector, circuit, q_input, None, n, q_aux, ancilla)
		sfGenerator(weightVector, circuit, q_input, None, n, q_aux, ancilla)

	elif circuitGeneratorOfUOperator == "encoding-weight":
		#inputVectorBinarized = thresholdBinarization(inputVector) # FOR REAL 0-1 INPUTS
		inputVectorBinarized = deterministicBinarization(inputVector)
		encodingGenerator2(weightVector, circuit, q_input)
		hsgsGenerator(inputVectorBinarized, circuit, q_input, n)
            
	elif circuitGeneratorOfUOperator == "encoding-input":
		encodingGenerator2(inputVector, circuit, q_input)
		weightVectorBinarized = deterministicBinarization(weightVector)
		hsgsGenerator(weightVectorBinarized, circuit, q_input, n)
		
	else:
		print("WARNING: nenhum neuronio valido selecionado")

	for i in range(n):
		circuit.h(q_input[i])
		circuit.x(q_input[i])
    
    
	#circuit.mcrx(math.pi, q_input, q_output[0])
	circuit.mcx(q_input, q_output[0], ancilla_qubits=None, mode='noancilla')

	circuit.measure(q_output, c_output)

	return circuit


def executeNeuron(neuronQuantumCircuit, simulator, threshold=None, nshots=8192):
    #from qiskit.tools.visualization import plot_histogram
	#neuronQuantumCircuit is the return of the function createNeuron
	#simulator function of a qiskit quantum simulator
	#expectedOutput is a Python List with expected value
		#e.g expectedOutput = [1] or expectedOutput = [0]
	#threshold is a real value between 0 or 1
	##this function returns the output 0 or 1 of the neuron depending of threshold value
    #nshots = 8192
    neuronOutput = int # FARIA MAI SENTIDO ISSO SER TRUE OU FALSE
    circuit = neuronQuantumCircuit
    job = execute(circuit, backend=simulator, shots=nshots)
    result = job.result()
    count = result.get_counts()

    
    # print(count)
    results1 = count.get('1') # Resultados que deram 1
    if str(type(results1)) == "<class 'NoneType'>": results1 = 0

    results0 = count.get('0') # Resultados que deram 0
    if str(type(results0)) == "<class 'NoneType'>": results0 = 0
       

    # Utilizando threshold
    if (threshold == None):
        return results1/nshots#, plot_histogram(count, title='Experiment')
    else:
        if (results1/nshots) >= threshold:
            neuronOutput = 1
        else:
            neuronOutput = 0 
        return neuronOutput


def trainNeuronDelta(nb_epochs, listOfInput, listOfExpectedOutput, circuitGeneratorOfUOperator, simulator, threshold, lr, memoryOfExecutions, binaryWeights=False, stochastic=True):
     # This function trains a quantum neuron
     import random
     y_train = listOfExpectedOutput
     x_train = listOfInput
     input_dim = x_train.shape[-1]
     data_len = len(y_train)
    
     w = np.random.uniform(-1, 1, input_dim)#np.random.rand(input_dim) # Real weights
     w = normalize(w)
    
     wB = makeBinarization(w, stochastic) # Binarization of Real weights
     maxHit=0
     for epoch in range(nb_epochs):
         y_pred = np.zeros(data_len)
         for i, x in enumerate(x_train):

             if binaryWeights:
                 circuit = createNeuron(x, wB, circuitGeneratorOfUOperator)     
                 out = executeNeuron(circuit, simulator, threshold)
             else:
                 circuit = createNeuron(x, w, circuitGeneratorOfUOperator)     
                 out = executeNeuron(circuit, simulator, threshold)

             #print(x,wB)
             
             if out > threshold: #   >0
                 y_pred[i] = 1
             #print(out, y_pred[i], y_train[i])
            
             if y_pred[i] != y_train[i]:
                 delta = y_train[i] - y_pred[i]

                 for j in range(input_dim):
                     w[j] = w[j] + (lr * delta * x_train[i][j])

                 wB = makeBinarization(w, stochastic)
            #  print(wB)

         hits = (y_train == y_pred).sum()
         maxHit = max(maxHit, (hits / data_len) * 100)
         print('Epoch {:d} accuracy: {:.2f} max acc {:.2f}'.format(epoch + 1, (hits / data_len) * 100, maxHit))


def trainNeuronPso(nb_epochs, listOfInput, listOfExpectedOutput, circuitGeneratorOfUOperator, simulator, threshold, lr, memoryOfExecutions, binaryWeights=False, stochastic=True):
    y_train = listOfExpectedOutput
    x_train = listOfInput
    input_dim = x_train.shape[-1]
    data_len = len(y_train)
    cost_func = []
    
    w = np.random.uniform(-1, 1, input_dim)#np.random.rand(input_dim) # Real weights
    w = normalize(w)
    
    wB = makeBinarization(w, stochastic) # Binarization of Real weights
    maxHit=0
    
    
    for epoch in range(nb_epochs):
        y_pred = np.zeros(data_len)
        for i, x in enumerate(x_train):

            
            if binaryWeights:
                circuit = createNeuron(x, wB, circuitGeneratorOfUOperator)     
                out = executeNeuron(circuit, simulator, threshold)
            else:
                circuit = createNeuron(x, w, circuitGeneratorOfUOperator)     
                out = executeNeuron(circuit, simulator, threshold)

            if out > threshold: #   >0
                y_pred[i] = 1

            if y_pred[i] != y_train[i]:

                PSO(costFuncPSO(w), w, bounds=[(0),(1)], num_particles=input_dim ,maxiter=100)
                wB = makeBinarization(w, stochastic)
            # print(wB)

        hits = (y_train == y_pred).sum()
        maxHit = max(maxHit, (hits / data_len) * 100)
        print('Epoch {:d} accuracy: {:.2f} max acc {:.2f}'.format(epoch + 1, (hits / data_len) * 100, maxHit))


def trainNeuron(method, nb_epochs, listOfInput, listOfExpectedOutput, circuitGeneratorOfUOperator, simulator, threshold, lr, memoryOfExecutions, binaryWeights=False, stochastic=True):
    if method == 'delta':
        trainNeuronDelta(nb_epochs, listOfInput, listOfExpectedOutput, circuitGeneratorOfUOperator, simulator, threshold, lr, memoryOfExecutions, binaryWeights=binaryWeights, stochastic=stochastic)
    elif method == 'pso':
        trainNeuronPso(nb_epochs, listOfInput, listOfExpectedOutput, circuitGeneratorOfUOperator, simulator, threshold, lr, memoryOfExecutions, binaryWeights=binaryWeights, stochastic=stochastic)
    else:
        print("ERROR: YOU NEED TO CHOOSE THE METHOD")

def runDataset(listOfInput, listOfExpectedOutput, weightVector, circuitGeneratorOfUOperator, simulator, threshold, memoryOfExecutions, printAcc=False, printErr=False):
	#listOfInput is a Python List 
		#e.g listOfInput= [[1,-1,1,1], [-1,-1,-1,1]]
	#listOfExpectedOutput is a Python List 
		#e.g listOfExpectedOutput= [[0], [1]]
	#weightVector is a Python list 
		#eg. weightVector=[1, 1, 1, -1]
	#circuitGeneratorOfUOperator is a function in python that will generate the Ui and Uw operators.
		#circuitGeneratorOfUOperator can be "hsgsGenerator", "SFGenerator", "EncodingGenerator"
	#simulator function of a qiskit quantum simulator
	#threshold is a real value between 0 or 1
	#memoryOfExecutions is a Python Dictionary with the executions and its results
		#the index is (inputVector, weightVector) and the content of the dic is the output of the neuron
		#e.g memoryOfExecutions = {([1,1,-1,1], [1,1,-1,1]: 0)}

	#for each input in listOfInput
		#generate the neuron circuit only if the configuration (input, weight) is not in Dic memoryOfExecutions
		#execute and save in the memoryOfExecutions


	##this function returns confusion matrix for this listOfInput and weightVector.

    #Variables initialization
    circuit = QuantumCircuit()
    truePositives = 0
    trueNegatives = 0
    falsePositives = 0
    falseNegatives = 0
    total = 0
    datasetSize = len(listOfInput)

    #Starting the run in the entire dataset
    for i in range(datasetSize):

        inputVector = listOfInput[i]
        theClass = listOfExpectedOutput[i]

        # ANTES DE EXECUTAR VERIFICAR NO DICIONARIO memoryOfExecutions
        chaveDictionary = (tuple(inputVector), tuple(weightVector))
        if (chaveDictionary in memoryOfExecutions):
            executionResult = memoryOfExecutions[chaveDictionary]
        else:
            circuit = createNeuron(inputVector, weightVector, circuitGeneratorOfUOperator)     
            executionResult = executeNeuron(circuit, simulator, threshold)
            memoryOfExecutions[chaveDictionary] = executionResult

        # Comparing the actual result with the expected result 
        if executionResult == 0: # neuronio deu como saida 0
            if theClass != 0: 
                falseNegatives = falseNegatives + 1
            else:
                trueNegatives = trueNegatives + 1
        else: # neuronio deu como saida 1
            if theClass != 1: 
                falsePositives = falsePositives + 1
            else:
                truePositives = truePositives + 1

    total = falseNegatives + falsePositives + trueNegatives + truePositives

    if printAcc:
        print("Acc:", (truePositives+trueNegatives)*100/total, '%')
    
    if printErr:
        print("Err:", (falsePositives+falseNegatives)*100/total, '%')

    return [[trueNegatives, falsePositives],[falseNegatives, truePositives]]
    
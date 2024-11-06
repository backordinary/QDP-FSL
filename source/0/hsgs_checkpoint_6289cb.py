# https://github.com/claudioalvesmonteiro/quantumML-health/blob/5e5feca1236041219c607e4a87b4117b811403aa/code/.ipynb_checkpoints/hsgs-checkpoint.py
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit
from qiskit.aqua import AquaError
from qiskit.compiler import transpile, assemble
from sympy.combinatorics.graycode import GrayCode
#from qiskit.circuit.library.standard_gates.multi_control_rotation_gates import _apply_mcu3_graycode, mcrx, mcrz
import numpy as np
import random
import math

def dec_to_bin(number, n): #função pra tranformar decimal em binário
    binary = bin(number)[2:].zfill(n)
    return binary

def find_position(base_state): #função para guardar as posições onde serão aplicadas as portas Z
    position = []
    for pos in range(len(base_state)):
        if base_state[pos] == '1':
            position.append(pos)
    return position

def atualizeVector(vector, n,  positions): #função que atualiza o vetor com os novos valores de amplitudes depois da aplicação do HSGS 
    for base_state in range(len(vector)):
        count = 0
        pos1_base_state = find_position(dec_to_bin(base_state, n))
        for pos in positions:
            if pos in pos1_base_state:
                count += 1
        if count == len(positions):
            vector[base_state] *= -1

def makeCZ(n, circuit, ctrls, qaux, qtarget): #função que aplica porta z multi-controlada entre 3 ou mais qubits
    m=0
    circuit.ccx(ctrls[0], ctrls[1], qaux[0])
    for m in range(2, len(ctrls)):
        circuit.ccx(ctrls[m], qaux[m-2], qaux[m-1])
    
    circuit.cz(qtarget, qaux[len(ctrls)-2])
    
    for m in range(len(ctrls)-1, 1, -1):
        circuit.ccx(ctrls[m], qaux[m-2], qaux[m-1])
    circuit.ccx(ctrls[0], ctrls[1], qaux[0])

    return circuit

def hsgsGenerator(inputVector, circuit, q_input, n, ancila=False):
	#inputVector is a Python list 
		#eg. inputVector=[1, -1, 1, 1]
	#circuit is the circuit where the HSGS will be put in
    #q_input ????
    #nSize is the input size
    #ancila esse parametro define se usamos registradores auxiliares para aplicar a porta multi-controlada Z ou não
    
	## this functions returns the quantum circuit that generates the quantum state whose amplitudes values are the values of inputVector using hsgsGenerator approach.

    if ancila == True:
        if n == 1:
            q_aux = QuantumRegister(n+1, 'q_aux')
        elif n == 2:
            q_aux = QuantumRegister(n, 'q_aux')
        else:
            q_aux = QuantumRegister(n-1, 'q_aux')
        circuit.add_register(q_aux)


    if(inputVector[0] == -1):
        inputVector = [indice * -1 for indice in inputVector]
    else:
        inputVector = [indice for indice in inputVector]
    
    outputVector = []
    for i in range(len(inputVector)):
        outputVector.append(1)
                    
    for p in range(1, n+1): 
        #laço pra pegar os p bits setado com 1 no estado base
        for base_state in range(2**n): 
            
            #percorre todos os estados da base
            positions = find_position(dec_to_bin(base_state,n)) 
            
            #encontrando as posições no estado da base onde o bit é '1'
            if len(positions) == p:
                if outputVector[base_state] != inputVector[base_state]: 
                    
                    #checa se o vetor final já está no estado desejado
                    circuit.barrier()
                    if len(positions) == 1: 
                        
                        #aplicar porta z simples para os estados da base com 1 unico qubit '1'
                        circuit.z(q_input[positions[0]])
                    elif len(positions) == 2: 
                        
                        #aplicar porta cz pas os estados da base com 2 qubits setados
                        circuit.cz(q_input[positions[0]], q_input[positions[1]])
                    else: 
                        
                        #aplicar uma porta control Z para 3 ou mais qubits
                        # setados com ajuda de qbits auxiliares
                        ctrls = [q_input[pos] for pos in positions[:len(positions)-1]]
                        target = q_input[positions[len(positions)-1]]
                        if ancila == True:
                            makeCZ(n, circuit, q_input, q_aux, target)
                        else:
                            circuit.mcrz(math.pi, ctrls, target)
                        
                    #atualiza vetor de saída
                    atualizeVector(outputVector, n, positions)
    return circuit
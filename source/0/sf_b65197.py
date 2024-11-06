# https://github.com/claudioalvesmonteiro/QuantumNeuronRealWeights/blob/ab295615a035d2d3b1cf574c64c15cc7c2c89d3f/sf.py
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit
from qiskit.aqua import AquaError
from qiskit.compiler import transpile, assemble
from sympy.combinatorics.graycode import GrayCode

#from qiskit.circuit.library.standard_gates.multi_control_rotation_gates import _apply_mcu3_graycode, mcrx, mcrz

import numpy as np
import random
import math

def decToBin(num, n):# Função para transformar um numero decimal numa string binária de tamanho n
    num_bin = bin(num)[2:].zfill(n)
    return num_bin

def findDec(input_vector, n): # Fução que pega as posições dos fatores -1 do vetor de entrada
    num_dec = []
    for i in range(0, len(input_vector)):
        if input_vector[i] == -1:
            num_dec.append(i)
    return num_dec

def findBin(num_dec, n): # Função que tranforma os numeros das posições em strings binarias
    num_bin = []
    for i in range(0, len(num_dec)):
        num_bin.append(decToBin(num_dec[i], n))
    return num_bin

def makeCZ(n, circuit, ctrls, q_aux, q_target): # Função que aplica uma porta Pauli-Z multi-controlada nos qubits de controle

    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])
    for m in range(2, len(ctrls)):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    
    circuit.cz(q_aux[n-2], q_target[0])
    
    for m in range(len(ctrls)-1, 1, -1):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])

    return circuit

def sfGenerator(inputVector, circuit, q_input, q_target, nSize, q_aux=None, ancila=True):# Função que aplica um Sign-Flip Block nos vetores de entradas e pesos
	#inputVector is a Python list 
		#eg. inputVector=[1, -1, 1, 1]
	#nSize is the input size

	## this functions returns the quantum circuit that generates the quantum state whose amplitudes values are the values of inputVector using the SFGenerator approach.
    """
    if ancila == True:
        q_aux = QuantumRegister(nSize-1, 'q_aux')
        circuit.add_register(q_aux)
    """
    positions = []
        
    # definindo as posições do vetor onde a amplitude é -1 
    # e tranformando os valores dessas posições em strings binárias
    # conseguindo os estados da base que precisarão ser modificados 
    positions = findDec(inputVector, nSize)
    pos_binary = findBin(positions, nSize)

    # laço para percorrer cada estado base em pos_binay
    for q_basis_state in pos_binary:
        # pegando cada posição da string do estado onde o bit é 0
        # aplicando uma porta Pauli-X para invertê-lo
        for indice_position in range(nSize):
            if q_basis_state[indice_position] == '0':
                circuit.x(q_input[indice_position])
        
        # aplicando porta Pauli-Z multi-controlada entres os qubits em q_input
        q_bits_controllers = [q_control for q_control in q_input[:nSize-1]]
        q_target = q_input[[nSize-1]]
        if ancila == True:
            makeCZ(nSize, circuit, q_input, q_aux, q_target)
        else:
            circuit.mcrz(math.pi, q_bits_controllers, q_target[0])
        
        # desfazendo a aplicação da porta Pauli-X nos mesmos qubits
        for indice_position in range(nSize):
            if q_basis_state[indice_position] == '0':
                circuit.x(q_input[indice_position])
    return circuit
# https://github.com/claudioalvesmonteiro/quantumML-health/blob/b8b63c9757f5114bacf8da34db0a789580c4b3ab/code/phaseEncoding2.py
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

def makePhaseEncodingAncilla(pi_angle, n, circuit, ctrls, q_aux, q_target, q_bits_controllers): 

    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])
    for m in range(2, len(ctrls)):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
        
    circuit.mcrz(pi_angle, q_bits_controllers, q_target[0])
    
    for m in range(len(ctrls)-1, 1, -1):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])

    return circuit


def phaseEncodingGenerator(inputVector, circuit, q_input, nSize, q_aux=None, ancila=False, weight=False):
    
    """
    PhaseEncoding Sign-Flip Block Algorithm
    
    inputVector is a Python list 
    nSize is the input size

    this functions returns the quantum circuit that generates the quantum state 
    whose amplitudes values are the values of inputVector using the SFGenerator approach.
    """
    
    positions = []
        
    # seleciona as posicoes do vetor 
    # e tranforma os valores dessas posicoes em strings binarias
    # conseguindo os estados da base que precisarao ser modificados 
    
    positions = list(range(len(inputVector)))
    pos_binary = findBin(positions, nSize)

    pi_angle_pos = 0
    # laço para percorrer cada estado base em pos_binay
    for q_basis_state in pos_binary:
        # pegando cada posição da string do estado onde o bit = 0
        # aplicando uma porta Pauli-X para inverte-lo
        for indice_position in range(nSize):
            if q_basis_state[indice_position] == '0':
                circuit.x(q_input[indice_position])
        
        # aplicando porta Pauli-Z multi-controlada entres os qubits em q_input
        q_bits_controllers = [q_control for q_control in q_input[:nSize-1]]
        q_target = q_input[[nSize-1]]
        if (nSize >2 ):
            circuit.mcrz(inputVector[pi_angle_pos], q_bits_controllers, q_target[0])
        else:
            circuit.rz(inputVector[pi_angle_pos], q_target[0])
            
            
        # desfazendo a aplicação da porta Pauli-X nos mesmos qubits
        for indice_position in range(nSize):
            if q_basis_state[indice_position] == '0':
                circuit.x(q_input[indice_position])
        pi_angle_pos+=1
        
    return circuit
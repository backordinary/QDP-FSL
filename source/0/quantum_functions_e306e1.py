# https://github.com/addie43110/qwalk-app-backend/blob/7f019e178cbe26beff7eb91904371ee414c1d002/api/quantum_functions.py
# importing Qiskit
import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from math import pi,log,ceil
import matplotlib as mpl
# import basic plot tools
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

mpl.use('Agg')

# QUANTUM HELPERS
def increment(n_adder):
    qc = QuantumCircuit(n_adder)
    for i in range(n_adder-2):
        qc.mcx(list(range(i+1, n_adder)),i)
    qc.cx(n_adder-1, n_adder-2)
    qc.x(n_adder-1)

    U_inc = qc.to_gate()
    U_inc.name = "U$_{inc}$"
    return U_inc

def decrement(n_sub):
    qc = QuantumCircuit(n_sub)
    qc.x(n_sub-1)
    qc.cx(n_sub-1, n_sub-2)
    for i in range(n_sub-3, -1, -1):
        qc.mcx(list(range(i+1, n_sub)),i)
               
    U_dec = qc.to_gate()
    U_dec.name = "U$_{dec}$"
    return U_dec

def shift1D(n_pos):
    n_dir = 1

    qr = QuantumRegister(n_dir+n_pos)
    q_dir = qr[:n_dir]
    q_pos = qr[n_dir:]
    qc = QuantumCircuit(qr)

    # if direction is 0 (LEFT)
    qc.x(q_dir)
    qc.append(decrement(n_pos).control(n_dir), q_dir+q_pos)
    qc.x(q_dir)
    # if direction is 1 (RIGHT)
    qc.append(increment(n_pos).control(n_dir), q_dir+q_pos)

    U_shift = qc.to_gate()
    U_shift.name = "U$_{shift}$"
    return U_shift

def shift2D(n_pos):
    n_dir = 2 
    HALF_N = ceil(n_pos/2)

    qr = QuantumRegister(n_dir+n_pos)
    q_dir = qr[:n_dir]
    q_pos = qr[n_dir:]
    qc = QuantumCircuit(qr)
    
    # if direction is 00 (RIGHT)
    qc.x(q_dir)
    qc.append(increment(HALF_N).control(2), q_dir+q_pos[HALF_N:])
    qc.x(q_dir)
    # if direction is 01 (DOWN)
    qc.x(q_dir[0])
    qc.append(increment(HALF_N).control(2), q_dir+q_pos[:-HALF_N])
    qc.x(q_dir[0])
    # if direction is 10 (LEFT)
    qc.x(q_dir[1])
    qc.append(decrement(HALF_N).control(2), q_dir+q_pos[HALF_N:])
    qc.x(q_dir[1])
    # if direction is 11 (UP)
    qc.append(decrement(HALF_N).control(2), q_dir+q_pos[:-HALF_N])
    
    
    U_shift = qc.to_gate()
    U_shift.name = "U$_{shift}$"
    return U_shift

def shift3D(n_pos):
    n_dir = 3
    LEN_ONE_SIDE_OF_CUBE = n_pos
    HALF_LEN = ceil(LEN_ONE_SIDE_OF_CUBE / 2)
    
    qr = QuantumRegister(n_dir+n_pos)
    q_dir = qr[:n_dir]
    q_pos = qr[n_dir:]
    
    qc = QuantumCircuit(qr)
    
    # if direction is 000 (RIGHT)
    qc.x(q_dir)
    qc.append(increment(HALF_LEN).control(3), q_dir+q_pos[-HALF_LEN:])
    qc.x(q_dir)
    
    # if direction is 001 (DOWN)
    qc.x(q_dir[0])
    qc.x(q_dir[1])
    qc.append(increment(HALF_LEN).control(3), q_dir+q_pos[HALF_LEN:-HALF_LEN])
    qc.x(q_dir[0])
    qc.x(q_dir[1])
    
    # if direction is 010 (LEFT)
    qc.x(q_dir[0])
    qc.x(q_dir[2])
    qc.append(decrement(HALF_LEN).control(3), q_dir+q_pos[-HALF_LEN:])
    qc.x(q_dir[0])
    qc.x(q_dir[2])
    
    # if direction is 011 (UP)
    qc.x(q_dir[0])
    qc.append(decrement(HALF_LEN).control(3), q_dir+q_pos[HALF_LEN:-HALF_LEN])
    qc.x(q_dir[0])
    
    # if direction is 100 (BACK)
    qc.x(q_dir[1])
    qc.x(q_dir[2])
    qc.append(increment(HALF_LEN).control(3), q_dir+q_pos[:HALF_LEN])
    qc.x(q_dir[1])
    qc.x(q_dir[2])
    
    # if direction is 101 (FORWARD)
    qc.x(q_dir[1])
    qc.append(decrement(HALF_LEN).control(3), q_dir+q_pos[:HALF_LEN])
    qc.x(q_dir[1])
    
    # if direction is 110
    
    # if direction is 111
    
    
    U_shift = qc.to_gate()
    U_shift.name = "U$_{shift}$"
    return U_shift

def round_remove_zeroes(np_dict):
    for a,d in np.ndenumerate(np_dict):
        for k,v in d.items():
            d[k] = round(v, 5)
    
    new_dict = {}
    for a,d in np.ndenumerate(np_dict):
        for k,v in d.items():
            if v:
                new_dict[k] = v
                
    return new_dict

def qwalk2D(dim, magnitude, iterations):
    n_dir = dim
    n_pos = magnitude
    qwalk_reg = QuantumRegister(n_dir+n_pos+1)

    # allocating qubits
    q_dir = qwalk_reg[:n_dir]
    q_pos = qwalk_reg[n_dir:n_dir+n_pos]
    q_anc = qwalk_reg[n_dir+n_pos]

    # lists of indices
    dir_ind = list(range(n_dir))
    pos_ind = list(range(n_dir+n_pos))[2:]
    anc_ind = n_dir+n_pos

    qwalk_circ = QuantumCircuit(qwalk_reg, ClassicalRegister(n_pos))

    # set up the initial position
        
    # set up the initial direction(s)
    for qubit in q_dir:
        qwalk_circ.h(qubit)

    states = []

    states.append(Statevector.from_instruction(qwalk_circ))
    for i in range(iterations):
        
        ''' Uncomment to add target state (Sticky walk) functionality'''
        #qwalk_circ.mcx(pos_ind, anc_ind) oracle, target is |11111..1>
        
        '''controlled shift where ancilla is control
        only shift if we are in a non-target state'''
        qwalk_circ.x(q_anc)

        if (dim == 1): # if 1D walk, use 1D shift
            qwalk_circ.append(shift1D(n_pos).control(1), [anc_ind]+dir_ind+pos_ind)
        elif (dim == 2): # if 2D walk, use 2D shift
            qwalk_circ.append(shift2D(n_pos).control(1), [anc_ind]+dir_ind+pos_ind)
        else:
            qwalk_circ.append(shift3D(n_pos).control(1), [anc_ind]+dir_ind+pos_ind)

        # add state to list
        states.append(Statevector.from_instruction(qwalk_circ))
        
        qwalk_circ.x(q_anc)
        
        '''Uncomment to add target state (Sticky walk) functionality'''
        #qwalk_circ.reset(q_anc)

        qwalk_circ.h(q_dir)

    return states

def create_plots2D(states):
    # PLOTTING FOR 8 x 8 GRID
    sum_data = np.zeros((8,8))
    meta_states = []
    counter = 0

    # states = states[len(states)-1:]

    for state in states:
        
        np_dict = round_remove_zeroes(np.array(state.probabilities_dict([7,6,5,4,3,2])))         
        #print("Position qubits:",np_dict)
        
        data = np.around(np.array(state.probabilities([7,6,5,4,3,2])), 5)
        data = np.reshape(data, (8,8))
        sum_data = sum_data + data
        meta_states.append(sum_data)
        d = state.probabilities_dict([7,6,5,4,3,2])
        
        # customizing plot
        plt.title("Current States")
        pixel_plot = plt.imshow(data, cmap='hot')
        if (counter==0):
            plt.colorbar(pixel_plot)
        plt.savefig('./images/dist'+str(counter)+'.png')
        counter +=1
# https://github.com/BensonZhou1991/W-state-teleportation/blob/1bc5ff89f2bd6654289a0873a58a067ab7af4ce8/w_state_teleportation.py
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:03:38 2021

@author: zxz58
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import BasicAer, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock import FakeVigo
device_backend = FakeVigo()

num_shot = 1024
intro_noise = 0

w_state = np.zeros(8)
w_state[1], w_state[2], w_state[4] = 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)

def BM(cir, q0, q1, c0, c1):
    cir.cx(q0, q1)
    cir.h(q0)
    cir.measure(q0, c0)
    cir.measure(q1, c1)

def W_teleportation():
    q = QuantumRegister(4)
    crz = ClassicalRegister(1) # and 2 classical bits
    crx = ClassicalRegister(1) # in 2 different registers
    cr_bm = ClassicalRegister(2)
    cr = ClassicalRegister(1)
    cr_final = ClassicalRegister(1)
    cir = QuantumCircuit(q, cr_bm, cr, cr_final)
    # ini W state
    ini_state = np.zeros(16)
    ini_state[2], ini_state[4], ini_state[8] = 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)
    cir.initialize(ini_state, q)
    # ini transmitted state
    cir.rx(np.pi/3, q[0])
    cir.barrier()
    # bell measurement
    cir.cx(q[0], q[1])
    cir.h(q[0])
    cir.measure(q[0], cr_bm[0])
    cir.measure(q[1], cr_bm[1])
    # measure
    cir.measure(q[2], cr)
    # recover
    cir.x(q[3]).c_if(cr_bm, 0)
    cir.x(q[3]).c_if(cr_bm, 1)
    cir.z(q[3]).c_if(cr_bm, 1)
    cir.z(q[3]).c_if(cr_bm, 3)
    # measure final qubit
    cir.measure(q[3], cr_final)
    return cir

def W_teleportation_multihop(n):
    # n: hop number
    num_q = 1 + 3 * n
    q = QuantumRegister(num_q, 'q')
    cir = QuantumCircuit(q)
    cr_bm_list = []
    for i in range(n):
        add_reg = ClassicalRegister(2, 'c_bm'+str(i))
        cr_bm_list.append(add_reg)
        cir.add_register(add_reg)
    cr = ClassicalRegister(n, 'c_cm') # successs measurement
    cir.add_register(cr)
    cr_final = ClassicalRegister(1, 'c_final') # final state measurement
    cir.add_register(cr_final)
    # ini transmitted state
    cir.h(q[0])
    cir.barrier()
    # ini W state
    for i in range(n):
        cir.initialize(w_state, q[3*i+1:3*i+4])
    # measurement
    for i in range(n):
        BM(cir, q[3*i], q[3*i+1], cr_bm_list[i][0], cr_bm_list[i][1])
        cir.measure(q[3*i+2], cr[i])
    # recover
    for i in range(n-1, -1, -1):
        cr_bm = cr_bm_list[i]
        cir.x(q[-1]).c_if(cr_bm, 0)
        cir.x(q[-1]).c_if(cr_bm, 1)
        cir.z(q[-1]).c_if(cr_bm, 1)
        cir.z(q[-1]).c_if(cr_bm, 3)
    # measure final qubit
    cir.measure(q[-1], cr_final)
    return cir

def SimW(cir):
    simulator_qasm = BasicAer.get_backend('qasm_simulator')
    noise_simulator = QasmSimulator.from_backend(device_backend)
    if intro_noise == 0:
        simulator = simulator_qasm
    else:
        simulator = noise_simulator
    result = execute(cir, simulator, shots=num_shot).result()
    counts = result.get_counts(cir)
    #plot_histogram(counts, title='Bell-State counts')
    return counts

def AnalyseRes(counts, n=1):
    succ_count = 0
    fail_count = 0
    i_state = 0
    i_succ = list(range(1, n+1))
    res_state = [0, 0] # to store the measurement res for the transmitted state
    for key in counts:
        key_list = []
        value = counts[key]
        # str to list
        for c in key:
            if c != ' ': key_list.append(int(c))
        # fail or success (results are all 0)?
        flag_succ = True
        for i in i_succ:
            if key_list[i] == 1:
                flag_succ = False
                break
        if flag_succ == True:
            succ_count += value
            # update measurement result for the transmitted state
            if key_list[i_state] == 0:
                res_state[0] += value
            else:
                res_state[1] += value
        else:
            fail_count += value
    print('number of hop is', n)
    print('success rate is', succ_count/(succ_count+fail_count))
    print('0 result probability is', res_state[0]/sum(res_state))
    print('measurement result for the transmitted state is', res_state)
    return succ_count, fail_count, res_state

if __name__ == '__main__':
    n = 2
    cir = W_teleportation_multihop(n)
    fig = (cir.draw(scale=0.7, filename=None, style=None, output='mpl', interactive=False, plot_barriers=1, reverse_bits=False))
    fig.savefig('cir.svg', format='svg', papertype='a4')
    counts = SimW(cir)
    succ_count, fail_count, res_state = AnalyseRes(counts, n)
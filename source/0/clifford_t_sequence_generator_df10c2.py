# https://github.com/tokarev-i-v/neuro-quantum-gate-simulator/blob/6a2fa6a6393ccb75d220a475a52f6c03c4ec5663/clifford_t_sequence_generator.py
import pandas as pd
import numpy as np
import os
from queue import Queue
import pdb
import argparse
from tqdm import tqdm
import ray

import time

from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
# from qiskit_textbook.tools import array_to_latex
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate, SGate, TGate, CXGate
from qiskit.visualization import plot_histogram
from collections import OrderedDict
from typing import List

from utils import cliffordt_actions_generator
from metrics import metrics

parser = argparse.ArgumentParser(description='This script generates quantum gates dataset using Qiskit.')

parser.add_argument('--output_type', default='hdf', type=str)

args = parser.parse_args()

ray.init()

# init unitary simulator
unitary_backend = Aer.get_backend('unitary_simulator')
# init unitary simulator
statevector_backend = Aer.get_backend('statevector_simulator')




"""
    ::arr:: list of objects
        {
            'gate': ZGate(), // specified gate
            'qubits': [1]    // spedified qubits
        }
"""
def generate_by_line(qc, item):
    gate_index = item['gate_index']
    init_unitary = execute(qc,unitary_backend).result().get_unitary()
    qc.unitary(item['gate'], item['qubits'])
    out_unitary = execute(qc,unitary_backend).result().get_unitary()
    df_item = {
        'gate_index': gate_index,
        'unitary': init_unitary,
        'next_unitary': out_unitary,
        'metric_value': metrics['trace_metric'](init_unitary, out_unitary)
    }
    return df_item
    # pdb.set_trace()


"""
    Function for generate  dataset of sequences of applyed gates.

    ::arr:: list of objects
        {
            'gate': ZGate(), // specified gate
            'qubits': [1]    // spedified qubits
            'gate_index': int
        }
"""
def generate_by_gates_array(qc, gates_array):
    init_unitary = execute(qc,unitary_backend).result().get_unitary()
    gates_indexes = []
    for item in gates_array:
        gate_index = item['gate_index']
        qc.unitary(item['gate'], item['qubits'])
        gates_indexes.append(gate_index)
    out_unitary = execute(qc,unitary_backend).result().get_unitary()
    df_item = {
        'gates_indexes': gates_indexes,
        'unitary': init_unitary,
        'next_unitary': out_unitary,
        'metric_value': metrics['trace_metric'](init_unitary, out_unitary)
    }
    return df_item
    # pdb.set_trace()



#generation algorithm
@ray.remote
def generate_datasets():
    # Count of qubits in quantumregister
    # We could create dataset with varying count of qubits
    qubits_count = 3
    # We got all qubits in 
    list_of_possible_actions = cliffordt_actions_generator(qubits_count)
    gates_lists = []
    #count of circuits to generate
    count_of_circuits = np.random.randint(100, 300)
    for i in range(count_of_circuits):
        le = []
        print(i)
        # length of circuit
        count_of_gates_in_circuit = np.random.randint(10, 300)
        for j in range(count_of_gates_in_circuit):
            ri = np.random.randint(0, len(list_of_possible_actions))
            le.append(list_of_possible_actions[ri])
        gates_lists.append(le)

    # inserting gates in their own queue
    queues = []
    for i in range(len(gates_lists)):
        q = []
        for item in gates_lists[i]:
            q.append(item)
        queues.append(q)

    pid = os.getpid()
    for q in tqdm(queues):
        df = []
        qc = QuantumCircuit(qubits_count)
        # get 
        df.append(generate_by_gates_array(qc, q))
        pd_df = pd.DataFrame(df)

        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.output_type == 'hdf':
            pd_df.to_hdf(f'sequence_datasets/quantum_dataset_{timestr}_{pid}output.h5', key='df')    
        else:
            pd_df.to_csv(f'sequence_datasets/quantum_dataset_{timestr}_{pid}output.csv')

returns = [generate_datasets.remote() for _ in range(11)]
ray.get(returns)
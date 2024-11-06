# https://github.com/tokarev-i-v/neuro-quantum-gate-simulator/blob/a7e22d5b1f377ce24dc36bd26051749a58af9615/generator.py
import pandas as pd
import numpy as np
from queue import Queue
import pdb
import argparse

import time
timestr = time.strftime("%Y-%m-%d_%H-%M-%S")

from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
# from qiskit_textbook.tools import array_to_latex

from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate
from qiskit.visualization import plot_histogram

from collections import OrderedDict
from typing import List

parser = argparse.ArgumentParser(description='This script generates quantum gates dataset using Qiskit.')

parser.add_argument('--output_type', default='hdf', type=str)

args = parser.parse_args()

gates_lists = [
    [
        {
            'gate': XGate(),
            'qubits': [0]
        },
        {
            'gate': XGate(),
            'qubits': [1]
        },
        {
            'gate': YGate(),
            'qubits': [0]
        },
        {
            'gate': YGate(),
            'qubits': [1]
        },
        {
            'gate': ZGate(),
            'qubits': [0]
        },
        {
            'gate': ZGate(),
            'qubits': [1]
        },
    ],
    [
        {
            'gate': XGate(),
            'qubits': [0]
        },
        {
            'gate': XGate(),
            'qubits': [1]
        },
        {
            'gate': YGate(),
            'qubits': [0]
        },
        {
            'gate': YGate(),
            'qubits': [1]
        },
        {
            'gate': ZGate(),
            'qubits': [0]
        },
        {
            'gate': ZGate(),
            'qubits': [1]
        },
    ],
    [
        {
            'gate': XGate(),
            'qubits': [0]
        },
        {
            'gate': XGate(),
            'qubits': [1]
        },
        {
            'gate': YGate(),
            'qubits': [0]
        },
        {
            'gate': YGate(),
            'qubits': [1]
        },
        {
            'gate': ZGate(),
            'qubits': [0]
        },
        {
            'gate': ZGate(),
            'qubits': [1]
        },
    ],    [
        {
            'gate': XGate(),
            'qubits': [0]
        },
        {
            'gate': XGate(),
            'qubits': [1]
        },
        {
            'gate': YGate(),
            'qubits': [0]
        },
        {
            'gate': YGate(),
            'qubits': [1]
        },
        {
            'gate': ZGate(),
            'qubits': [0]
        },
        {
            'gate': ZGate(),
            'qubits': [1]
        },
    ],
]

df = {
    "gate": [], # Name of Gate
    "qubits": [], # string of qubits
    "statevector": [], # Quantum state 
    "next_statevector": [],
    "unitary": [],
    "next_unitary": []
}

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
def generate_by_line(arr, qubits_count):
    qc = QuantumCircuit(qubits_count)
    for item in arr:
        # pdb.set_trace()
        qubits = "".join(str(ch) for ch in item['qubits'])
        gate = item['gate'].name
        init_statevector = execute(qc,statevector_backend).result().get_statevector()
        init_unitary = execute(qc,unitary_backend).result().get_unitary()
        qc.unitary(item['gate'], item['qubits'])
        out_statevector = execute(qc,statevector_backend).result().get_statevector()
        out_unitary = execute(qc,unitary_backend).result().get_unitary()
        df['gate'].append(gate)
        df['qubits'].append(qubits)
        df['statevector'].append(init_statevector)
        df['next_statevector'].append(out_statevector)
        df['unitary'].append(init_unitary)
        df['next_unitary'].append(out_unitary)
        # pdb.set_trace()
        



# inserting gates in their own queue
queues = []
end_queues = []
for i in range(len(gates_lists)):
    q = []
    for item in gates_lists[i]:
        q.append(item)
    queues.append(q)
    end_queues.append([])


# Count of qubits in quantumregister
qubits_count = 2

# generation algorithm
while True:
    if len(queues[0]) == 0:
        break
    i = len(queues) - 1
    while True:
        # если указатель на последнем индексе!!!
        # то создаём линии
        if i == len(queues) - 1:
            for j in range(len(queues[i])):
                line = [item[0] for item in queues]
                generate_by_line(line, qubits_count)
                # print(line)
                # t += 1
                # print(t)
                queues[i].append(queues[i].pop(0))
                # pdb.set_trace()
            i -= 1
            continue
        # if current is one of average index (!= 0 && != - 1)
        elif i != 0:
            #
            # if len(queues[i]) != 0:
            end_queues[i].append(queues[i].pop())
            if len(queues[i]) == 0:
                for _ in range(len(end_queues[i])):
                    queues[i].append(end_queues[i].pop())
                # come to i - 1 index to process it
                i -= 1
            else:
                # come to creating new lines
                break
            # else:
            #     # come to creating new lines
            #     break
        else:
            queues[0].pop()
            break

pd_df = pd.DataFrame(df)
if args.output_type == 'hdf':
    pd_df.to_hdf(f'quantum_dataset_{timestr}_output.h5', key='df')    
else:
    pd_df.to_csv(f'quantum_dataset_{timestr}_output.csv')
# https://github.com/snuphya/GST_qiskit/blob/9ad98227e7e476158802e32e7663c8f64d59585d/utils/parse.py
# this file is deprecated
# gates : ["{}", "Gxpi2", ...]
import pygsti
from qiskit import IBMQ
import numpy as np
from qiskit import *
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.qobj.utils import MeasLevel

class Parse:
    def __init__(self):
        self.fill_ones()
        self.ds2 = pygsti.io.load_dataset("data/dataset.txt")

    def fill_ones(self):
        with open('data/dataset.txt') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if len(line)<3:
                    continue
                elif line[0]=='#':
                    continue
                else:
                    lines[idx] = line[:-2] + '1'
        
        with open('data/dataset.txt', 'w') as f:
            f.write('\n'.join(lines) + '\n')

    def write_dataset(self, meas_dict, meas_key):
        with open('data/dataset.txt') as f:
            lines = f.readlines()
            meas_dict_idx = 0
            for idx, line in enumerate(lines):
                if len(line)<3:
                    continue
                elif line[0]=='#':
                    continue
                else:
                    while lines[idx][-1] != ')':
                        lines[idx] = lines[idx][:-1]
                    for key in meas_key:
                        lines[idx] += '\t'
                        lines[idx] += str(meas_dict[meas_dict_idx][key]) if key in meas_dict[meas_dict_idx] else str(0)
                    meas_dict_idx += 1 
                    
        with open('data/dataset.txt', 'w') as f:
            f.write('\n'.join(lines) + '\n')
            

    def parse_datasets(self, ):
        circs = []
        for key in self.ds2.keys():
            circs.append(self.get_circ_from_layer(key.layertup))
        # ds2.keys()[0].layertup[2].name
        # ds2.keys()[0].layertup[2].qubits
        return circs


    def get_circ_from_layer(self, layer):
        circ = QuantumCircuit(1)
        for gate in layer:
            if gate==None:
                circ.id()
            
            # Be careful of rotation direction
            elif gate.name=='Gxpi2':
                circ.rx(np.pi/2,gate.qubits)
            elif gate.name=='Gypi2':
                circ.ry(np.pi/2,gate.qubits)
            else:
                NotImplementedError("## Gate {} is not implemented in get circ_from layer".format(gate.name))
        circ.measure_all()
        return circ
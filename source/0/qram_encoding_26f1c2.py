# https://github.com/sami-b95/Qiskit-Toolbox/blob/14a378f4bb3594a441ae0a2f7e0af59dd4c07adf/Encoding/QRAM-Encoding/QRAM_encoding.py
from qiskit import QuantumRegister, QuantumCircuit
import os
os.chdir('../../Preprocessing')
from Classical_data_preparation import lineic_preprocessing, euclidean_norm
from Classical_boolean_tests import is_log_concave_encoding_compatible
os.chdir('../BitStringTools')
from Bit_string_tools import to_bin, x_gates_region
os.chdir('../Encoding/QRAM-Encoding')
import numpy as np
from qiskit.aqua.circuits.gates import mcry


def qRAM_encoding_angles(distribution, n_qubits):
    distribution = np.array(distribution)
    size = distribution.size
    distribution = lineic_preprocessing(distribution, vertical=False)
    if is_log_concave_encoding_compatible(distribution, n_qubits) == True:
        distribution = np.sqrt(distribution)
        angles = {}
        for step in range(n_qubits):
            inter = 2**(step)
            limit_region = int(size / (inter * 2)) * np.arange(inter * 2 + 1)
            inter_list = []
            for region in range(inter):
                if  euclidean_norm(distribution[limit_region[2 * region]:limit_region[2 * region + 1]]) == 0:
                    inter_list.append(np.pi/2)
                else:
                    inter_list.append(np.arctan2(euclidean_norm(distribution[limit_region[2 * region + 1]:limit_region[2 * region + 2]]), euclidean_norm(distribution[limit_region[2 * region]:limit_region[2 * region + 1]])))
            angles[step] = inter_list
        return angles
    else:
        raise NameError('The distribution is not compatible with the number of qubits or is not normalized or has negative values.')
       
        

def qRAM_encoding(distribution, n_qubits):
    theta = qRAM_encoding_angles(distribution, n_qubits)
    qubits = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qubits)   
    circuit.u3(2 * theta[0][0], 0, 0, qubits[n_qubits - 1])
    for step in range(n_qubits - 1):
        step = step + 1
        control_qubits = list(map(lambda x: qubits[n_qubits - x - 1], range(step)))
        for region in range(2 ** step):
            circuit = x_gates_region(circuit, qubits, to_bin(region, step))
            circuit.mcry(- 2 * theta[step][region], control_qubits, qubits[n_qubits - step - 1], None, 'noancilla')
            circuit = x_gates_region(circuit, qubits, to_bin(region, step))        
    return circuit

#Faire l'inverse de la qRAM encoding

    




# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/Encoding/QRAM-Encoding/qRAMEncodingGate.py
from qiskit import QuantumRegister
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.ry import RYGate
from AbstractGates.qiwiGate import qiwiGate
from BitStringTools.Bit_string_tools import XRegionGate
from Preprocessing.Classical_data_preparation import lineic_preprocessing, euclidean_norm
from Preprocessing.Classical_boolean_tests import is_log_concave_encoding_compatible
import numpy as np


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

                
class qRAMEncodingGate(qiwiGate):
    """qRAM Encoding gate."""
    
    def __init__(self, num_qubits, distribution, least_significant_bit_first=True):
        self.num_qubits = num_qubits
        self.least_significant_bit_first = least_significant_bit_first
        self.distribution = distribution
        super().__init__(name=f"qRAM Encoding", num_qubits=num_qubits, params=[], least_significant_bit_first=least_significant_bit_first)
        
    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits)
        if self.least_significant_bit_first:
            q = q[::-1]
        theta = qRAM_encoding_angles(self.distribution, self.num_qubits)
        definition.append((U3Gate(2 * theta[0][0], 0, 0), [q[self.num_qubits - 1]], []))
        for step in range(self.num_qubits - 1):
            step = step + 1
            ctrl_q = list(map(lambda x: q[self.num_qubits - x - 1], range(step)))
            for region in range(2 ** step):
                definition.append((XRegionGate(self.num_qubits, region), q, []))
                definition.append((RYGate(- 2 * theta[step][region]).control(len(ctrl_q)), ctrl_q + [q[self.num_qubits - step - 1]], []))
                definition.append((XRegionGate(self.num_qubits, region), q, []))
        if self.least_significant_bit_first:
            q = q[::-1]
        self.definition = definition
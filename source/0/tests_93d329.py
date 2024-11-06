# https://github.com/Traivok/quantum-computing-assignment/blob/6b9909815160543c47e9ed6709c1794bda314de8/tests.py
import unittest
import numpy as np
from encoding import Encoding
from main import gen_angles, gen_circuit

from qiskit import Aer, execute, QuantumCircuit

state_default = list(map(lambda a: np.sqrt(a), [.03, .07, .15, .05, .1, .3, .2, .1]))

def normalize(v):
    return v / np.linalg.norm(v)

class test(unittest.TestCase):
    def test1(self):
        assert(np.allclose(gen_angles(state_default), [1.98, 1.91, 1.43, 1.98, 1.05, 2.09, 1.23], rtol=.01))
    def test2(self):
        backend = Aer.get_backend('statevector_simulator')
        get_result = lambda circuit: execute(circuit, backend).result().get_statevector()

        r1 = get_result( Encoding(np.array(state_default), 'dc_amplitude_encoding').qcircuit )
        r2 = get_result( gen_circuit( gen_angles(state_default) ) )

        assert(np.allclose(r2, r1, rtol=0.01))

# https://github.com/Quantumyilmaz/Quantum-Algorithms/blob/d33c7b16ea8f946bb50fb1f7cf53d487ad99ca79/utils/arithmetic/qft_arithmetic.py
# Author: Ahmet Ege Yilmaz
# Year: 2021
# QFT Addition & Subtraction

import numpy as np

from utils.gates import AddGate,SubtractGate
from utils.misc import encode_integer, get_counts, counts_to_integer, prepare_integers

from qiskit import QuantumCircuit

"""
qft_arithmetic_to_integer(qft_adder(20,5,n=10,to_gate=False))
"""

def qft_adder(add_to,add_this, n=10, to_gate=True, draw_barriers=False):
    a , b = add_to , add_this
    assert n >= len(np.binary_repr(a+b)),len(np.binary_repr(a+b))

    qc = QuantumCircuit(2*n,name= f"add {add_this}" + bool(add_to)*f" to {add_to}")

    AddTo = encode_integer(add_to)
    qc.compose(AddTo.to_gate(),range(AddTo.num_qubits),inplace=True)

    qc.compose(AddGate(n=n,add_this=add_this),inplace=True)

    return qc.to_gate() if to_gate else qc


"""
qft_arithmetic_to_integer(qft_subtractor(12,5,n=4,to_gate=False))
"""

def qft_subtractor(subtract_from, subtract_this, n=10, to_gate=True):

    assert subtract_from >= subtract_this

    qc = QuantumCircuit(2*n,name= f"subtract {subtract_this}" + bool(subtract_from)*f" from {subtract_from}")

    qc.compose(prepare_integers(n=n,integers=[subtract_from],to_gate=True),inplace=True)
    qc.compose(SubtractGate(n,subtract_this),inplace=True)

    return qc.to_gate() if to_gate else qc


def qft_arithmetic_to_integer(circ):
    return counts_to_integer(get_counts(circ.copy(),circ.qubits[:circ.num_qubits//2]))

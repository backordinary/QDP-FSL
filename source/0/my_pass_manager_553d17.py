# https://github.com/farnaznouraei/ADD-Qiskit/blob/9654b67f8a764876aea6af4b30794eae2470d970/qiskit/transpiler/my_pass_manager.py
import qiskit
from qiskit import *
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasicSwap
from qiskit.extensions.standard import SwapGate
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import Decompose
from qiskit.transpiler.passes import SetLayout


def my_pass_manager(coupling_map,initial_layout):

    _given_layout = SetLayout(initial_layout);
    pm = PassManager([Unroller(['u1','u2','u3','cx']),_given_layout,BasicSwap(coupling_map),Decompose(SwapGate)]);
    return pm





# https://github.com/vietphamngoc/QPAC/blob/e64d34d10d3e9dca993717707c5da5e8d9d70a93/qaa.py
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator

from oracle import Oracle
from tnn import TNN


def get_diffusion_operator(ora: Oracle, tun_net: TNN):
    """
    Function to generate the diffusion operator corresponding to the query oracle and the tunable network in its current state.

    Arguments:
        - ora: Oracle, the query oracle
        - tun_net: TNN, the tunable neural network

    Returns:
        The quantum gate representing the diffusion operator
    """
    n = tun_net.dim
    qc = QuantumCircuit(n+2)
    # Chi_g
    qc.cz(n, n+1)
    # A^-1
    qc.cry(-2*np.arcsin(1/np.sqrt(5)), n, n+1)
    qc.append(tun_net.network, range(n+1))
    qc.append(ora.inv_gate, range(n+1))
    # -Chi_0
    mat = -np.eye(2**(n+2))
    mat[0,0] = 1
    op = Operator(mat)
    qc.unitary(op, range(n+2), label="Chi_0")
    # A
    qc.append(ora.gate, range(n+1))
    qc.append(tun_net.network, range(n+1))
    qc.cry(2*np.arcsin(1/np.sqrt(5)), n, n+1)
    return(qc.to_gate(label="Diffusion"))
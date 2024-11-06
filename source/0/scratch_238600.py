# https://github.com/cpbunker/learn_qiskit/blob/a1c81db8f61f1f00cc710b6f05ce17fdb623e6f2/mining/scratch.py
'''
https://github.com/cpbunker/learn/qiskit

Use variational quantum eigensolver (VQE) to solve a binary linear
optimization (BLO) problem
'''

import utils
import visuals

import numpy as np
import matplotlib.pyplot as plt

import qiskit
import qiskit.quantum_info as qi

# implement the BLO as a Hamiltonian
weights = [-1.0,-0.8];
ham = utils.BLO_to_ham(weights);

# construct an ansatz for the Hamiltonian
from qiskit.circuit.library import TwoLocal
num_qubits = len(weights);
wf_ansatz = TwoLocal(num_qubits,"ry","cz");
print(wf_ansatz.decompose().draw());

# use VQE to find the best ansatz
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP
myVQE = VQE(qiskit.primitives.Estimator(),wf_ansatz,SLSQP());
result = myVQE.compute_minimum_eigenvalue(ham);
wf_answer = result.optimal_circuit;
wf_answer = wf_answer.bind_parameters(result.optimal_parameters);
#print(wf_answer.decompose().draw());
#visuals.circuit_counts(wf_answer);

# the gd state energy
for myqc in [utils.str_to_qc("0"*wf_answer.num_qubits),utils.str_to_qc("1"*wf_answer.num_qubits),wf_answer]:
    psi = qi.Statevector(myqc);
    Egd = psi.evolve(ham);
    Egd = psi.inner(Egd);
    print("\n",psi.to_dict(), "\n",Egd)

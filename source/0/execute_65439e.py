# https://github.com/muehlhausen/vqls-bachelor-thesis/blob/40f22453a1832be0487dbdf7ed44e82107919d87/Simulation/execute.py
# all libraries used by some part of the VQLS-implementation
from qiskit import (
    QuantumCircuit, QuantumRegister, ClassicalRegister,
    Aer, execute, transpile, assemble
    )
from qiskit.circuit import Gate, Instruction
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import ZGate, YGate, XGate, IGate

from scipy.optimize import (
                    minimize, basinhopping, differential_evolution,
                    shgo, dual_annealing
                    )

import random

import numpy as np
import cmath

from typing import List, Set, Dict, Tuple, Optional, Union


# import the params object of the GlobalParameters class
# this provides the parameters used to desribed and model
# the problem the minimizer is supposed to use.
from GlobalParameters import params

# import the vqls algorithm and corresponding code
from vqls import (
    generate_ansatz,
    hadamard_test,
    calculate_beta,
    calculate_delta,
    calculate_local_cost_function,
    minimize_local_cost_function,
    postCorrection,
    _format_alpha,
    _calculate_expectationValue_HadamardTest,
    _U_primitive
    )

# The user input for the VQLS-algorithm has to be given
# when params is initialized within GlobalParameters.py

# The decomposition for $A$ has to be manually
# inserted into the code of
# the class GlobalParameters.

print(
    "This program will execute a simulation of the VQLS-algorithm "
    + "with 4 qubits, 4 layers in the Ansatz and a single Id-gate acting"
    + " on the second qubit.\n"
    + "To simulate another problem, one can either alter _U_primitive "
    + "in vqls.py to change |x_0>, GlobalParameters.py to change A "
    + "or its decomposition respectively."
)

# Executing the VQLS-algorithm
alpha_min = minimize_local_cost_function(params.method_minimization)


"""
Circuit with the $\vec{alpha}$ generated by the minimizer.
"""

# Create a circuit for the vqls-result
qr_min = QuantumRegister(params.n_qubits)
circ_min = QuantumCircuit(qr_min)

# generate $V(\vec{alpha})$ and copy $A$
ansatz = generate_ansatz(alpha_min).to_gate()
A_copy = params.A.copy()
if isinstance(params.A, Operator):
    A_copy = A_copy.to_instruction()

# apply $V(\vec{alpha})$ and $A$ to the circuit
# this results in a state that is approximately $\ket{b}$
circ_min.append(ansatz, qr_min)
circ_min.append(A_copy, qr_min)

# apply post correction to fix for sign errors and a "mirroring"
# of the result
circ_min = postCorrection(circ_min)


"""
Reference circuit based on the definition of $\ket{b}$.
"""

circ_ref = _U_primitive()

"""
Simulate both circuits.
"""
# the minimizations result
backend = Aer.get_backend(
          'statevector_simulator')
t_circ = transpile(circ_min, backend)
qobj = assemble(t_circ)
job = backend.run(qobj)
result = job.result()


print(
    "This is the result of the simulation.\n"
    + "Reminder: 4 qubits and an Id-gate on the second qubit."
    + "|x_0> was defined by Hadamard gates acting on qubits 0 and 3.\n"
    + "The return value of the minimizer (alpha_min):\n"
    + str(alpha_min)
    + "\nThe resulting statevector for a circuit to which "
    + "V(alpha_min) and A and the post correction were applied:\n"
    + str(result.get_statevector())
)

t_circ = transpile(circ_ref, backend)
qobj = assemble(t_circ)
job = backend.run(qobj)
result = job.result()

print(
    "And this is the statevector for the reference circuit: A |x_0>\n"
    + str(result.get_statevector())
)

print("these were Id gates and in U y on 0 and 1")

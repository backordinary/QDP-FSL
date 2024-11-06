# https://github.com/ctuning/ck-qiskit/blob/9a606a1ad9497d142486b788074827a3e6aeab11/soft/template.qiskit.ansatz/python_code/reduced_universal5/custom_ansatz.py
#!/usr/bin/env python3

import numpy as np
import qiskit

num_params = 5      # make sure you set this correctly to the number of parameters used by the ansatz


# Derived from a depth 2 HW efficient ansatz circuit, by finding rotations that
# were multiples of pi. This gets to -1.13 (slightly worse than a true solution)
# and so might be a good ansatz for people to improve upon.

def reduced_HW_efficient_ansatz(current_params):
    q = qiskit.QuantumRegister(2, "q")
    qc = qiskit.QuantumCircuit(q, qiskit.ClassicalRegister(2, "c"))

    qc.x(q[0])
    qc.rz(current_params[0], q[0])
    qc.y(q[1])

    qc.cx(q[0], q[1])

    qc.rz(current_params[1], q[0])
    qc.rz(current_params[2], q[1])
    qc.y(q[0])
    qc.y(q[1])

    # no CNOT, works without it.

    qc.rx(current_params[3], q[0])
    qc.rz(current_params[4], q[1])
    qc.z(q[0])
    qc.y(q[1])

    qc.cx(q[0], q[1])

    return qc

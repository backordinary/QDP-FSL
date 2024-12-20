# https://github.com/jwoehr/qisjob/blob/b4a7fecb791cc43e3fc09aeafc2cf8a4bd1127ff/share/qc_examples/google_quantum_supremacy.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Quantum Supremacy Circuit
https://quantum-circuit.com/app_details/about/3SQoZgKdJ6oqaX5gq
https://quantum-circuit.com/api/get/circuit/GuavdH643Wbeusuww?format=qiskit
Created on Sat Nov  2 05:39:58 2019

Intended to be loaded and run from qisjob
@author: jax
"""
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qc = QuantumCircuit()

q = QuantumRegister(5, 'q')
c = ClassicalRegister(5, 'c')

qc.add_register(q)
qc.add_register(c)

qc.rx(np.pi / 2, q[0])
qc.ry(np.pi / 2, q[1])
qc.rx(np.pi / 2, q[2])
qc.ry(np.pi / 2, q[3])
qc.rx(np.pi / 2, q[4])
qc.u3(0, 0, -(np.pi / 2), q[1])
qc.u3(0, 0, -(np.pi / 2), q[2])
qc.cz(q[2], q[1])
qc.swap(q[2], q[1])
qc.cu1(np.pi / 6, q[2], q[1])
qc.ry(np.pi / 2, q[0])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[1])
qc.ry(np.pi / 2, q[2])
qc.rx(np.pi / 2, q[3])
qc.ry(np.pi / 2, q[4])
qc.u3(0, 0, -(np.pi / 2), q[2])
qc.u3(0, 0, -(np.pi / 2), q[3])
qc.cz(q[2], q[3])
qc.swap(q[2], q[3])
qc.cu1(np.pi / 6, q[2], q[3])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[0])
qc.rx(np.pi / 2, q[1])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[2])
qc.rx(np.pi / 2, q[3])
qc.ry(np.pi / 2, q[4])
qc.u3(0, 0, -(np.pi / 2), q[0])
qc.u3(0, 0, -(np.pi / 2), q[2])
qc.cz(q[2], q[0])
qc.swap(q[2], q[0])
qc.cu1(np.pi / 6, q[2], q[0])
qc.rx(np.pi / 2, q[0])
qc.ry(np.pi / 2, q[1])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[2])
qc.rx(np.pi / 2, q[3])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[4])
qc.u3(0, 0, -(np.pi / 2), q[2])
qc.u3(0, 0, -(np.pi / 2), q[4])
qc.cz(q[2], q[4])
qc.swap(q[2], q[4])
qc.cu1(np.pi / 6, q[2], q[4])
qc.rx(np.pi / 2, q[0])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[1])
qc.ry(np.pi / 2, q[2])
qc.rx(np.pi / 2, q[3])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[4])
qc.u3(0, 0, -(np.pi / 2), q[0])
qc.u3(0, 0, -(np.pi / 2), q[2])
qc.cz(q[2], q[0])
qc.swap(q[2], q[0])
qc.cu1(np.pi / 6, q[2], q[0])
qc.ry(np.pi / 2, q[0])
qc.rx(np.pi / 2, q[1])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[2])
qc.rx(np.pi / 2, q[3])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[4])
qc.u3(0, 0, -(np.pi / 2), q[2])
qc.u3(0, 0, -(np.pi / 2), q[4])
qc.cz(q[2], q[4])
qc.swap(q[2], q[4])
qc.cu1(np.pi / 6, q[2], q[4])
qc.ry(np.pi / 2, q[0])
qc.rx(np.pi / 2, q[1])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[2])
qc.rx(np.pi / 2, q[3])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[4])
qc.u3(0, 0, -(np.pi / 2), q[1])
qc.u3(0, 0, -(np.pi / 2), q[2])
qc.cz(q[2], q[1])
qc.swap(q[2], q[1])
qc.cu1(np.pi / 6, q[2], q[1])
qc.rx(np.pi / 2, q[0])
qc.ry(np.pi / 2, q[1])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[2])
qc.ry(np.pi / 2, q[3])
qc.u3(np.pi / 2, -(np.pi / 4), np.pi / 4, q[4])
qc.u3(0, 0, -(np.pi / 2), q[2])
qc.u3(0, 0, -(np.pi / 2), q[3])
qc.cz(q[2], q[3])
qc.swap(q[2], q[3])
qc.cu1(np.pi / 6, q[2], q[3])
qc.measure(q[0], c[0])
qc.measure(q[1], c[1])
qc.measure(q[2], c[2])
qc.measure(q[3], c[3])
qc.measure(q[4], c[4])

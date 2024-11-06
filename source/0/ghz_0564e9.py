# https://github.com/niefermar/CuanticaProgramacion/blob/cf066149b4bd769673e83fd774792e9965e5dbc0/examples/python/ghz.py
# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
GHZ state example illustrating mapping onto the backend.

Note: if you have only cloned the Qiskit repository but not
used `pip install`, the examples only work from the root directory.
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import register, execute
import Qconfig

###############################################################
# Set the backend name and coupling map.
###############################################################
backend = "ibmq_5_tenerife"
coupling_map = [[0, 1],
                [0, 2],
                [1, 2],
                [3, 2],
                [3, 4],
                [4, 2]]

###############################################################
# Make a quantum circuit for the GHZ state.
###############################################################
q = QuantumRegister(5, "q")
c = ClassicalRegister(5, "c")
qc = QuantumCircuit(q, c, name='ghz')

# Create a GHZ state
qc.h(q[0])
qc.cx(q[0], q[0+1])
qc.cx(q[1], q[1+1])
qc.cx(q[2], q[2+1])
qc.cx(q[3], q[3+1])
# Insert a barrier before measurement
qc.barrier()
qc.measure(q[0], c[0])
qc.measure(q[1], c[1])
qc.measure(q[2], c[2])
qc.measure(q[3], c[3])
qc.measure(q[4], c[4])

###############################################################
# Set up the API and execute the program.
###############################################################
register(Qconfig.APItoken, Qconfig.config["url"])

# First version: no mapping
print("no mapping, execute on simulator")
job = execute(qc, backend='ibmq_qasm_simulator', coupling_map=None, shots=1024)
result = job.result()
print(result)
print(result.get_counts("ghz"))

# Second version: map to ibmq_5_tenerife coupling graph and simulate online
print("map to %s, execute on online simulator" % backend)
job = execute(qc, backend='ibmq_qasm_simulator', coupling_map=coupling_map, shots=1024)
result = job.result()
print(result)
print(result.get_counts("ghz"))

# Third version: map to ibmq_5_tenerife coupling graph and simulate locally
print("map to %s, execute on local simulator" % backend)
job = execute(qc, backend='local_qasm_simulator', coupling_map=coupling_map, shots=1024)
result = job.result()
print(result)
print(result.get_counts("ghz"))

# Fourth version: map to ibmq_5_tenerife coupling graph and run on ibmq_5_tenerife
print("map to %s, execute on device" % backend)
job = execute(qc, backend=backend, shots=1024)
result = job.result()
print(result)
print(result.get_counts("ghz"))

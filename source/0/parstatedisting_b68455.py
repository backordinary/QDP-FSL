# https://github.com/klinus9542/UnitaryDistIBMQ/blob/f0a4e3f40929546e4bc1c5b0e4f20b36518900d1/ParStateDisting.py
# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
ParDist state example illustrating mapping onto the backend.
"""

import sys
import os
import numpy
import math

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumProgram

import Qconfig

###############################################################
# Set the backend name and coupling map.
###############################################################
backend = "ibmqx4"
coupling_map = {0: [],
                1: [0],
                2: [0, 1, 4],
                3: [2, 4],
                4: []}



backendB="ibmqx2"
coupling_mapB = {0: [1, 2], 1: [2], 3: [2, 4], 4: [2]}


###############################################################
# Make a quantum program for the ParDist state.
###############################################################
QPS_SPECS = {
    "circuits": [{
        "name": "ParDist",
        "quantum_registers": [{
            "name": "q",
            "size": 5
        }],
        "classical_registers": [
            {"name": "c",
             "size": 5}
        ]}]
}

qp = QuantumProgram(specs=QPS_SPECS)
qc = qp.get_circuit("ParDist")
q = qp.get_quantum_register("q")
c = qp.get_classical_register("c")

#Creat rand number
RndNum=numpy.random.randint(0,2)

#Mix state produced
#if RndNum==1:
#qc.x(q[0])

#Init state
qc.h(q[0])
qc.barrier()


qc.cx(q[0],q[1])
qc.barrier()

qc.u3(1.231,0,0,q[0])  #This is X on first qubit
qc.barrier()

RndNum=0
if RndNum==0:
    qc.u1(2/3*math.pi,q[0])      #This is U on first qubit
    qc.u1(2/3*math.pi,q[1])      #This is U on second qubit
qc.barrier()

#qc.u3(2.186276035465283,2*math.pi, 2.879793265790644,q[0]) #This is R1
qc.u3(2.186276035465283,6.544984694978735, math.pi,q[0]) #This is R1'
#qc.u3(0.955316618124509,math.pi,4.450589592585540,q[0]) #This is R2
#qc.u3(0.955316618124509,4.974188368183840,2*math.pi,q[0]) #This is R2'
qc.barrier()

qc.u3(0.955316618124509,2*math.pi,math.pi,q[1]) #This is R


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
qp.set_api(Qconfig.APItoken, Qconfig.config["url"])
#
# # First version: no mapping
# print("no mapping, simulator")
# result = qp.execute(["ParDist"], backend='ibmqx_qasm_simulator',
#                     coupling_map=None, shots=3072)
# print(result)
# print(result.get_counts("ParDist"))
#
# # Second version: map to qx2 coupling graph and simulate
# print("map to %s, simulator" % backend)
# result = qp.execute(["ParDist"], backend='ibmqx_qasm_simulator',
#                     coupling_map=coupling_map, shots=3072)
# print(result)
# answer=result.get_counts("ParDist")
# print(answer)

#for key in answer.keys():
#    aa=[int(key[-1]),int(key[-2]),int(key[-3]),int(key[-4]),int(key[-5])]
#for key in answer:
#    print(key, 'corresponds', answer[key])
 #   print(int(key[-1]),int(key[-2]),int(key[-3]),int(key[-4]),int(key[-5]),':',answer[key])


# Third version: map to qx4 coupling graph and simulate locally
#print("map to %s, local qasm simulator" % backend)
#result = qp.execute(["ParDist"], backend='local_qasm_simulator',
#                    coupling_map=coupling_map, shots=1024)
#print(result)
#print(result.get_counts("ParDist"))

# Fourth version: map to qx2 coupling graph and run on qx2
print("map to %s, backend" % backend)
result = qp.execute(["ParDist"], backend=backend,
                    coupling_map=coupling_map, shots=1024, timeout=300)
print(result)
print(result.get_counts("ParDist"))
print(result.get_ran_qasm("ParDist"))
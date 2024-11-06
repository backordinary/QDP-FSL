# https://github.com/indisoluble/qasm-simulator-cpp-debian-pkg/blob/5fb81719defacf8731b96c74d934d7bf9191afa8/qiskit-sdk-py-0.5.1/examples/python/ghz.py
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
GHZ state example illustrating mapping onto the backend.

Note: if you have only cloned the QISKit repository but not
used `pip install`, the examples only work from the root directory.
"""

from qiskit import QuantumProgram
import Qconfig

###############################################################
# Set the backend name and coupling map.
###############################################################
backend = "ibmqx2"
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}

###############################################################
# Make a quantum program for the GHZ state.
###############################################################
QPS_SPECS = {
    "circuits": [{
        "name": "ghz",
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
qc = qp.get_circuit("ghz")
q = qp.get_quantum_register("q")
c = qp.get_classical_register("c")

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
qp.set_api(Qconfig.APItoken, Qconfig.config["url"])

# First version: no mapping
print("no mapping, simulator")
result = qp.execute(["ghz"], backend='ibmq_qasm_simulator',
                    coupling_map=None, shots=1024)
print(result)
print(result.get_counts("ghz"))

# Second version: map to qx2 coupling graph and simulate
print("map to %s, simulator" % backend)
result = qp.execute(["ghz"], backend='ibmq_qasm_simulator',
                    coupling_map=coupling_map, shots=1024)
print(result)
print(result.get_counts("ghz"))

# Third version: map to qx2 coupling graph and simulate locally
print("map to %s, local qasm simulator" % backend)
result = qp.execute(["ghz"], backend='local_qasm_simulator',
                    coupling_map=coupling_map, shots=1024)
print(result)
print(result.get_counts("ghz"))

# Fourth version: map to qx2 coupling graph and run on qx2
print("map to %s, backend" % backend)
result = qp.execute(["ghz"], backend=backend,
                    coupling_map=coupling_map, shots=1024, timeout=120)
print(result)
print(result.get_counts("ghz"))

# https://github.com/lanl/Quantum-Telecloning/blob/783d854083cc01f52cdcf3f2124bb2b8cd1cffa0/run_experiments/run_IBMQ_deferred_measurement_AAPCCC.py
"""
© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
"""

import sys
sys.path.append("../")
from utils import *
from create_telecloning_circuits import *
from qiskit import *
import copy
from qiskit.transpiler import CouplingMap
import numpy as np

OPT_LEVEL = 3

device_name = "ibmq_montreal"

subg = [6, 1, 4, 7, 10, 12, 15]
#subg = [17, 12, 15, 18, 21, 23, 24]
#subg = [9, 3, 5, 8, 11, 14, 16]
#subg = [20, 14, 16, 19, 22, 25, 26]

n = len(subg)

IBMQ.load_account()
provider = IBMQ.get_provider(hub='', group='', project='')
backend = provider.get_backend(device_name)
coupling_map = CouplingMap(backend.configuration().coupling_map)

N_shots = 30000
N_Angle_slices = 17

#Create a name for the qubit set used when storing results
subgraph_name = "_"
for a in subg:
	subgraph_name += str(a)+"_"
subgraph_name += "_"

print("Initial layout:", subgraph_name)

def get_transpiled_circuits(angle_ry, angle_rz):
	qc_tc, clone_indices, _ = construct_deferred_measurement_AAPCCC(angle_ry, angle_rz)
	tomography_circuits = parallel_qubit_state_tomography_general(clone_indices, qc_tc, angle_ry, angle_rz)
	transpiled = transpile(copy.deepcopy(tomography_circuits), coupling_map=coupling_map, optimization_level=OPT_LEVEL, basis_gates=["x", "sx", "cx", "rz"], initial_layout=subg)
	return transpiled

angles_ry = np.linspace(0, math.pi, num=N_Angle_slices)
angles_rz = np.linspace(0, 2*math.pi, num=N_Angle_slices)

compiled_circuits = []

for angle_ry in angles_ry:
	for angle_rz in angles_rz:
		circuits = get_transpiled_circuits(float(angle_ry), float(angle_rz))
		compiled_circuits += circuits

print("total circuits before error mitigation =", len(compiled_circuits))
qubits_to_measure = extract_qubit_indices_which_are_measured(compiled_circuits)
measurement_err_mitigation_circuits = generate_transpiled_measurement_error_mitigation_circuits(qubits_to_measure, coupling_map)
compiled_circuits = measurement_err_mitigation_circuits+compiled_circuits
print("total circuits=", len(compiled_circuits))

job = backend.run(compiled_circuits, shots=N_shots)
id = job.job_id()
print(id)

job_filename = "AAPCCC_"+device_name+subgraph_name+"_"+str(N_shots)+"_"+str(N_Angle_slices)+"_OPT_"+str(OPT_LEVEL)+".txt"
file = open("job_ids_deferred_measurement/"+job_filename, "w")
file.write(str(id))
file.close()

# https://github.com/lanl/Quantum-Telecloning/blob/783d854083cc01f52cdcf3f2124bb2b8cd1cffa0/classical_simulation_code/run_deferred_measurement_local_APCC.py
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
import numpy as np
from qiskit.ignis.verification.tomography import StateTomographyFitter
from qiskit.quantum_info import state_fidelity
import copy

backend = Aer.get_backend('qasm_simulator')

N_shots = 30000

def run(angle_ry, angle_rz):
	print("Ry angle =", angle_ry)
	print("Rz angle =", angle_rz)
	qc_tc_original, clone_indices, rho = construct_deferred_measurement_APCC(angle_ry, angle_rz)
	qc_tc = copy.deepcopy(qc_tc_original)
	
	tomography_circuits = parallel_qubit_state_tomography_general(clone_indices, copy.deepcopy(qc_tc), angle_ry, angle_rz)
	job = backend.run(tomography_circuits, shots=N_shots)
	result = job.result()
	
	counts = result.get_counts("Y_"+str(angle_ry)+"_"+str(angle_rz))
	counts_clone1_Y = get_single_qubit_measurements_from_parallel_results(counts, clone_indices[0])
	counts_clone2_Y = get_single_qubit_measurements_from_parallel_results(counts, clone_indices[1])
	
	counts = result.get_counts("X_"+str(angle_ry)+"_"+str(angle_rz))
	counts_clone1_X = get_single_qubit_measurements_from_parallel_results(counts, clone_indices[0])
	counts_clone2_X = get_single_qubit_measurements_from_parallel_results(counts, clone_indices[1])
	
	counts = result.get_counts("Z_"+str(angle_ry)+"_"+str(angle_rz))
	counts_clone1_Z = get_single_qubit_measurements_from_parallel_results(counts, clone_indices[0])
	counts_clone2_Z = get_single_qubit_measurements_from_parallel_results(counts, clone_indices[1])
	
	clone1_counts = {"X": counts_clone1_X, "Y": counts_clone1_Y, "Z": counts_clone1_Z}
	clone2_counts = {"X": counts_clone2_X, "Y": counts_clone2_Y, "Z": counts_clone2_Z}	
	
	clone1_tomography_circuits = state_tomography_clone1(clone_indices, copy.deepcopy(qc_tc_original))
	
	fitter = StateTomographyFitter(clone1_counts, clone1_tomography_circuits)
	fitted = fitter.fit(method='lstsq')
	Fidelity_clone1 = state_fidelity(rho, fitted)
	print("Clone 1 fidelity:", Fidelity_clone1)
	
	fitter = StateTomographyFitter(clone2_counts, clone1_tomography_circuits)
	fitted = fitter.fit(method='lstsq')
	Fidelity_clone2 = state_fidelity(rho, fitted)
	print("Clone 2 fidelity:", Fidelity_clone2)


angles = np.linspace(0, 2*math.pi, num=100)

for angle in angles:
	for angle2 in angles:
		run(float(angle), float(angle2))
		print("***")

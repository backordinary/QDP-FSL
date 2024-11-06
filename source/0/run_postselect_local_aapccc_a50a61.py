# https://github.com/lanl/Quantum-Telecloning/blob/783d854083cc01f52cdcf3f2124bb2b8cd1cffa0/classical_simulation_code/run_postselect_local_AAPCCC.py
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
	X_counts_clone1 = {"0": 0, "1": 0}
	Y_counts_clone1 = {"0": 0, "1": 0}
	Z_counts_clone1 = {"0": 0, "1": 0}
	
	X_counts_clone2 = {"0": 0, "1": 0}
	Y_counts_clone2 = {"0": 0, "1": 0}
	Z_counts_clone2 = {"0": 0, "1": 0}
	
	X_counts_clone3 = {"0": 0, "1": 0}
	Y_counts_clone3 = {"0": 0, "1": 0}
	Z_counts_clone3 = {"0": 0, "1": 0}
	classical_post_selected_states = ["00", "11", "01", "10"]
	for classical_state in classical_post_selected_states:
		if classical_state == "11":
			qc_tc_original, clone_indices, rho, post_select_indices = construct_AAPCCC_postselect_11(angle_ry, angle_rz)
		elif classical_state == "10":
			qc_tc_original, clone_indices, rho, post_select_indices = construct_AAPCCC_postselect_10(angle_ry, angle_rz)
		elif classical_state == "01":
			qc_tc_original, clone_indices, rho, post_select_indices = construct_AAPCCC_postselect_01(angle_ry, angle_rz)
		elif classical_state == "00":
			qc_tc_original, clone_indices, rho, post_select_indices = construct_AAPCCC_postselect_00(angle_ry, angle_rz)
		else:
			print("Problem!")
			sys.exit()
		tomography_circuits = parallel_qubit_state_tomography_general(clone_indices, copy.deepcopy(qc_tc_original), angle_ry, angle_rz)
		tomography_circuits = add_measurements_for_postselect_circuits_for_tomography_circuits(copy.deepcopy(tomography_circuits), post_select_indices)
		job = backend.run(tomography_circuits, shots=N_shots)
		result = job.result()
		
		counts = result.get_counts("Y_"+str(angle_ry)+"_"+str(angle_rz))
		c_1_Y = get_single_qubit_measurements_from_parallel_results_with_postselection(counts, clone_indices[0], post_select_indices)
		c_2_Y = get_single_qubit_measurements_from_parallel_results_with_postselection(counts, clone_indices[1], post_select_indices)
		c_3_Y = get_single_qubit_measurements_from_parallel_results_with_postselection(counts, clone_indices[2], post_select_indices)
		Y_counts_clone1 = merge_dictionaries(c_1_Y, Y_counts_clone1)
		Y_counts_clone2 = merge_dictionaries(c_2_Y, Y_counts_clone2)
		Y_counts_clone3 = merge_dictionaries(c_3_Y, Y_counts_clone3)
		
		counts = result.get_counts("X_"+str(angle_ry)+"_"+str(angle_rz))
		c_1_X = get_single_qubit_measurements_from_parallel_results_with_postselection(counts, clone_indices[0], post_select_indices)
		c_2_X = get_single_qubit_measurements_from_parallel_results_with_postselection(counts, clone_indices[1], post_select_indices)
		c_3_X = get_single_qubit_measurements_from_parallel_results_with_postselection(counts, clone_indices[2], post_select_indices)
		X_counts_clone1 = merge_dictionaries(c_1_X, X_counts_clone1)
		X_counts_clone2 = merge_dictionaries(c_2_X, X_counts_clone2)
		X_counts_clone3 = merge_dictionaries(c_3_X, X_counts_clone3)
		
		counts = result.get_counts("Z_"+str(angle_ry)+"_"+str(angle_rz))
		c_1_Z = get_single_qubit_measurements_from_parallel_results_with_postselection(counts, clone_indices[0], post_select_indices)
		c_2_Z = get_single_qubit_measurements_from_parallel_results_with_postselection(counts, clone_indices[1], post_select_indices)
		c_3_Z = get_single_qubit_measurements_from_parallel_results_with_postselection(counts, clone_indices[2], post_select_indices)
		Z_counts_clone1 = merge_dictionaries(c_1_Z, Z_counts_clone1)
		Z_counts_clone2 = merge_dictionaries(c_2_Z, Z_counts_clone2)
		Z_counts_clone3 = merge_dictionaries(c_3_Z, Z_counts_clone3)
		
	clone1_counts = {"X": X_counts_clone1, "Y": Y_counts_clone1, "Z": Z_counts_clone1}
	clone2_counts = {"X": X_counts_clone2, "Y": Y_counts_clone2, "Z": Z_counts_clone2}
	clone3_counts = {"X": X_counts_clone3, "Y": Y_counts_clone3, "Z": Z_counts_clone3}
	
	qc_00_circuit, _, _, _ = construct_AAPCCC_postselect_00(angle_ry, angle_rz)
	clone1_tomography_circuits = get_state_tomography_clone1_for_post_selection(clone_indices, copy.deepcopy(qc_00_circuit))
	
	fitter = StateTomographyFitter(clone1_counts, clone1_tomography_circuits)
	fitted = fitter.fit(method='lstsq')
	Fidelity_clone1 = state_fidelity(rho, fitted)
	print("Clone 1 fidelity:", Fidelity_clone1)
	
	fitter = StateTomographyFitter(clone2_counts, clone1_tomography_circuits)
	fitted = fitter.fit(method='lstsq')
	Fidelity_clone2 = state_fidelity(rho, fitted)
	print("Clone 2 fidelity:", Fidelity_clone2)
	
	fitter = StateTomographyFitter(clone3_counts, clone1_tomography_circuits)
	fitted = fitter.fit(method='lstsq')
	Fidelity_clone3 = state_fidelity(rho, fitted)
	print("Clone 3 fidelity:", Fidelity_clone3)

angles = np.linspace(0, 2*math.pi, num=100)

for angle in angles:
	for angle2 in angles:
		run(angle, angle2)
		print("***")

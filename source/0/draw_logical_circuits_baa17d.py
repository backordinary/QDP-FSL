# https://github.com/lanl/Quantum-Telecloning/blob/783d854083cc01f52cdcf3f2124bb2b8cd1cffa0/draw_logical_circuits.py
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

from create_telecloning_circuits import *
import math
from qiskit import QuantumCircuit
from utils import add_measurements_for_postselect_circuits

#Message qubit state
angle_y = math.pi
angle_z = math.pi

###############
# Post select 11
###############
qc, _, _, post_select_indices = construct_AAPCCC_postselect_11(angle_y, angle_z)
qc = add_measurements_for_postselect_circuits(qc, post_select_indices)
qc.draw(output='mpl', filename="circuit_drawings/post_selection_11_AAPCCC.pdf", idle_wires=False, fold=100)

qc, _, _, post_select_indices = construct_PCCC_postselect_11(angle_y, angle_z)
qc = add_measurements_for_postselect_circuits(qc, post_select_indices)
qc.draw(output='mpl', filename="circuit_drawings/post_selection_11_PCCC.pdf", idle_wires=False, fold=100)

qc, _, _, post_select_indices = construct_APCC_postselect_11(angle_y, angle_z)
qc = add_measurements_for_postselect_circuits(qc, post_select_indices)
qc.draw(output='mpl', filename="circuit_drawings/post_selection_11_APCC.pdf", idle_wires=False, fold=100)

qc, _, _, post_select_indices = construct_PCC_postselect_11(angle_y, angle_z)
qc = add_measurements_for_postselect_circuits(qc, post_select_indices)
qc.draw(output='mpl', filename="circuit_drawings/post_selection_11_PCC.pdf", idle_wires=False, fold=100)


##############
# Deferred measurement:
##############
qc, indices, rho = construct_deferred_measurement_APCC(angle_y, angle_z)
qc.draw(output='mpl', filename="circuit_drawings/deferred_measurement_APCC.pdf", idle_wires=False, fold=100)

qc, indices, rho = construct_deferred_measurement_PCC(angle_y, angle_z)
qc.draw(output='mpl', filename="circuit_drawings/deferred_measurement_PCC.pdf", idle_wires=False, fold=100)

qc, indices, rho = construct_deferred_measurement_AAPCCC(angle_y, angle_z)
qc.draw(output='mpl', filename="circuit_drawings/deferred_measurement_AAPCCC.pdf", idle_wires=False, fold=100)

qc, indices, rho = construct_deferred_measurement_PCCC(angle_y, angle_z)
qc.draw(output='mpl', filename="circuit_drawings/deferred_measurement_PCCC.pdf", idle_wires=False, fold=100)


##############
# If statements
##############
qc, _, _ = construct_PCC_Honeywell_QASM_circuit_Qiskit_version(angle_y, angle_z, True)
qc.draw(output='mpl', filename="circuit_drawings/Quantinuum_feed_forward_PCC.pdf", idle_wires=False, fold=100)
print("PCC", dict(qc.count_ops()))

qc, _, _ = construct_APCC_Honeywell_QASM_circuit_Qiskit_version(angle_y, angle_z, True)
qc.draw(output='mpl', filename="circuit_drawings/Quantinuum_feed_forward_APCC.pdf", idle_wires=False, fold=100)
print("APCC", dict(qc.count_ops()))

qc, _, _ = construct_PCCC_Honeywell_QASM_circuit_Qiskit_version(angle_y, angle_z, True)
qc.draw(output='mpl', filename="circuit_drawings/Quantinuum_feed_forward_PCCC.pdf", idle_wires=False, fold=100)
print("PCCC", dict(qc.count_ops()))

qc, _, _ = construct_AAPCCC_Honeywell_QASM_circuit_Qiskit_version(angle_y, angle_z, True)
qc.draw(output='mpl', filename="circuit_drawings/Quantinuum_feed_forward_AAPCCC.pdf", idle_wires=False, fold=100)
print("AAPCCC", dict(qc.count_ops()))

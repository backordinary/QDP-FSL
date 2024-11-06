# https://github.com/mtweiden/hybrid_partitioning/blob/01a8a070737b53c56906be78356f6a8c27c7b918/mapping.py
# Qiskit dependencies
from __future__ import annotations
import pickle
from typing import Any

from qiskit import QuantumCircuit
import qiskit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import (
	SabreLayout, SabreSwap, LookaheadSwap
)
from qiskit.transpiler.passmanager import PassManager
from pytket.qasm import circuit_to_qasm_str, circuit_from_qasm
from pytket.routing import Architecture, route
from pytket.transform import Transform
from pytket.passes import FullPeepholeOptimise
# Standard dependencies
from re import match, findall
from sys import argv
from random import shuffle
# Project dependiences
from coupling import get_coupling_map


def find_num_qudits(
	input_qasm_file : str,
) -> int:
	with open(input_qasm_file, 'r') as f:
		for line in f:
			if match("qreg ", line):
				num_logical = int(findall("\d+",line)[0])
				break
	return num_logical


def manual_layout(input_qasm_file, remapping_file, output_qasm_file, options):
	with open(remapping_file, "rb") as f:
		remapping = pickle.load(f)
	with open(input_qasm_file, 'r') as in_qasm:
		with open(output_qasm_file, 'w') as out_qasm:
			for line in in_qasm:
				if match('qreg q\[\d+\];', line):
					out_qasm.write('qreg q[%d];\n' %(options["num_p"]))
				else:
					out_qasm.write(format(line, remapping))
	return remapping 



def do_layout(input_qasm_file, coupling_map_file, output_qasm_file):

	# Gather circuit data
	circ = QuantumCircuit.from_qasm_file(input_qasm_file)
	num_logical_qubits = circ.width()
	(num_q, coupling_graph) = get_coupling_map(coupling_map_file, 
		num_logical_qubits, make_coupling_map_flag=True)

	# Set up Passes
	#seed = 42
	layout = SabreLayout(
		coupling_map=CouplingMap(list(coupling_graph)),
		#seed=seed,
		routing_pass=None,
		max_iterations=35
	)

	pass_man = PassManager([layout])

	# Create circuit with new layout and layout dictionary
	new_circ = pass_man.run(circ)
	qiskit_map = layout.property_set['layout'].get_virtual_bits()
	l2p_map = {l.index: qiskit_map[l] for l in qiskit_map}
	# Qiskit doesn't use physical numbering in the qasm() method, so parse the
	# qasm file and do the mapping here.
	with open(input_qasm_file, 'r') as in_qasm:
		with open(output_qasm_file, 'w') as out_qasm:
			for line in in_qasm:
				if match('qreg q\[\d+\];', line):
					out_qasm.write('qreg q[%d];\n' %(num_q))
				else:
					out_qasm.write(format(line, l2p_map))
	# Return the logical to physical mapping so it can be saved
	return l2p_map


def do_routing(
	input_qasm_file, 
	coupling_map_file, 
	output_qasm_file, 
	options=None,
	do_layout=False
) -> bool:
	router = options["router"] if options is not None else "qiskit"

	# Gather circuit data
	(num_q, coupling_graph) = get_coupling_map(coupling_map_file)

	if "qiskit" in router:
		if "lookahead" in router:
			try:
				circ = QuantumCircuit.from_qasm_file(input_qasm_file)
				if num_q >= circ.width():
					# Set up Passes
					#seed = 42
					coup_map = CouplingMap(list(coupling_graph))
					routing = LookaheadSwap(
						coupling_map=coup_map,
						search_depth=5,
						search_width=5,
						#seed=seed
					)
					pass_man = PassManager([routing])
					new_circ = pass_man.run(circ)
					new_qasm = new_circ.qasm()
				else:
					print("  WARNING: Router could not handle this coupling graph")
					return False
			except (qiskit.transpiler.exceptions.CouplingError,
				qiskit.transpiler.exceptions.TranspilerError):
				print("  WARNING: Router could not handle this coupling graph")
				return False
		else:
			try:
				circ = QuantumCircuit.from_qasm_file(input_qasm_file)
				# Post routing optimization
				#circ = qiskit.transpile(
				#	circ , 
				#	optimization_level=3, 
				#	basis_gates=["cx", "u3"],
				#	#coupling_map=CouplingMap(list(coupling_graph)),
				#)
				#from qiskit.transpiler.passes.basis.unroller import Unroller
				#from qiskit.transpiler.passes.basis.unroll_custom_definitions import UnrollCustomDefinitions
				##unroll = Unroller(basis=['cx', 'u3'])
				#unroll = UnrollCustomDefinitions(basis=['cx', 'u3'])
				#circ = unroll(circ)
				if num_q >= circ.width():
					# Set up Passes
					#seed = 42
					coup_map = CouplingMap(list(coupling_graph))
					routing = SabreSwap(
						coupling_map=coup_map,
						heuristic='lookahead',
						#seed=seed
					)
					pass_man = PassManager([routing])
					new_circ = pass_man.run(circ)
					new_qasm = new_circ.qasm()
					#qiskit_map = routing.property_set['final_layout'].get_virtual_bits()
					#l2p_map = {l.index: qiskit_map[l] for l in qiskit_map}
				else:
					print("  WARNING: Router could not handle this coupling graph")
					return False
			except (qiskit.transpiler.exceptions.CouplingError,
				qiskit.transpiler.exceptions.TranspilerError):
				print("  WARNING: Router could not handle this coupling graph")
				return False
	elif router == "pytket":
		circ = circuit_from_qasm(input_qasm_file)
		arch = Architecture(connections=list(coupling_graph))
		routed_circ = route(circuit=circ, architecture=arch)
		Transform.DecomposeBRIDGE().apply(routed_circ)
		new_qasm = circuit_to_qasm_str(routed_circ)

	
	with open(output_qasm_file, 'w') as out_qasm:
		out_qasm.write(new_qasm)
	return True
	#return l2p_map


def dummy_layout(input_qasm_file, coupling_map_file, output_qasm_file):
	# Make sure to expand the circuit to the number of physical qubits
	circ = QuantumCircuit.from_qasm_file(input_qasm_file)
	num_logical_qubits = circ.width()
	(num_q, _) = get_coupling_map(coupling_map_file, 
		num_logical_qubits, make_coupling_map_flag=True)

	with open(input_qasm_file, 'r') as in_qasm:
		with open(output_qasm_file, 'w') as out_qasm:
			for line in in_qasm:
				if match('qreg q\[\d+\];', line):
					out_qasm.write('qreg q[%d];\n' %(num_q))
				else:
					out_qasm.write(line)


def random_layout_dict(num_qudits : int) -> dict[int,int]:
	qudits = [x for x in range(num_qudits)]
	shuffle(qudits)
	return {k:qudits[k] for k in range(len(qudits))}


def random_layout(input_qasm_file, coupling_map_file, output_qasm_file):
	# Make a random layout 
	circ = QuantumCircuit.from_qasm_file(input_qasm_file)
	num_logical_qubits = circ.width()
	(num_q, _) = get_coupling_map(coupling_map_file, 
		num_logical_qubits, make_coupling_map_flag=True)
	layout_dict = random_layout_dict(num_q)
	# WRITE NEW QASM
	with open(input_qasm_file, 'r') as in_qasm:
		with open(output_qasm_file, 'w') as out_qasm:
			for line in in_qasm:
				if match('qreg q\[\d+\];', line):
					out_qasm.write(f'qreg q[{num_q}];\n')
				else:
					out_qasm.write(format(line, layout_dict))


def dummy_routing(input_qasm_file, coupling_map_file, output_qasm_file):
	with open(input_qasm_file, 'r') as in_qasm:
		with open(output_qasm_file, 'w') as out_qasm:
			out_qasm.write(in_qasm.read())


def dummy_synthesis(
	block_name : str,
	options : dict[str, Any],
) -> None:
	synth_dir = f"{options['synthesis_dir']}/{block_name}"
	block_path = f"{options['partition_dir']}/{block_name}"

	with open(f"{block_path}.qasm", "r") as f:
		block_qasm = f.read()

	with open(f"{synth_dir}.qasm", "w") as f:
		f.write(block_qasm)


def format(input_line, layout_map) -> str:
    """
    Returns new qasm line with physical qubit numbering.

    Args:
        input_line (str): Original qasm string.

        layout_map (dict): Logical to physical mapping.
    
    Returns:
        qasm_str (str): Newly mapped qasm string.

    Note:
        Assumes there is a single register named 'q', replaces that name with 
        'physical'.
    """
    # Find all instances of qubit reference
    if match('qreg', input_line):
        return input_line.replace('q[', 'physical[')

    qasm_str = input_line

    log_qubits = list(findall('q\[\d+\]', input_line))
    for lq in log_qubits:
        log_number = int(findall('\d+', lq)[0])
        replacement = 'physical[' + str(layout_map[log_number]) + ']'
        qasm_str = qasm_str.replace(lq, replacement)
        
    return qasm_str.replace('physical[', 'q[')


if __name__ == "__main__":
	if len(argv) != 3:
		print("Missing arguments")
		print(">>> python3 mapping <qasm file> <linear | mesh | alltoall>")
		quit
	else:
		do_layout(argv[1], argv[2])

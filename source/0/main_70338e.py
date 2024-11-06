# https://github.com/AntonyMei/SabreTest/blob/9e93e8d5dd88135e1b6a07fedd7543d0de31652d/main.py
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout

# identical to IBM Q20 Tokyo
coupling = [
    # rows
    [0, 1], [1, 2], [2, 3], [3, 4],
    [5, 6], [6, 7], [7, 8], [8, 9],
    [10, 11], [11, 12], [12, 13], [13, 14],
    [15, 16], [16, 17], [17, 18], [18, 19],
    # cols
    [0, 5], [5, 10], [10, 15],
    [1, 6], [6, 11], [11, 16],
    [2, 7], [7, 12], [12, 17],
    [3, 8], [8, 13], [13, 18],
    [4, 9], [9, 14], [14, 19],
    # crossings
    [1, 7], [2, 6],
    [3, 9], [4, 8],
    [5, 11], [6, 10],
    [8, 12], [7, 13],
    [11, 17], [12, 16],
    [13, 19], [14, 18]
]
coupling_map = CouplingMap(couplinglist=coupling)

# parse qasm file into a circuit
circuit = QuantumCircuit(20)
with open('sabre.qasm') as file:
    # omit the header
    file.readline()
    file.readline()
    line = file.readline()
    num_qubits = int(line.split(' ')[1].split(']')[0].split('[')[1])
    print(num_qubits)
    # parse the rest
    line = file.readline()
    while line != '':
        # add to circuit
        arg_list = line.split(' ')
        if arg_list[0] == '':
            arg_list = arg_list[1:]
        if len(arg_list) == 3:
            # two qubits gate
            qubit1 = int(arg_list[1].split(']')[0].split('[')[1])
            qubit2 = int(arg_list[2].split(']')[0].split('[')[1])
            circuit.cx(qubit1, qubit2)
        elif len(arg_list) == 2:
            # single qubit gate
            qubit1 = int(arg_list[1].split(']')[0].split('[')[1])
            circuit.h(qubit1)
        else:
            assert False
        # read another line
        line = file.readline()

# run sabre
layout_parser = SabreLayout(coupling_map=coupling_map)
pass_manager = PassManager(layout_parser)
basic_circ = pass_manager.run(circuit)

# print mapping
layout = layout_parser.property_set["layout"]
logical2physical = []
for logical_idx in range(num_qubits):
    for physical_idx in range(20):
        if layout[physical_idx].index == logical_idx:
            logical2physical.append(physical_idx)
print(logical2physical)

# https://github.com/StefanHaslhofer/QuantumCircuitMapper/blob/17127cdbd5381dbe2953714ad2f5389831c10378/helper.py
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout


# author: Elias Foramitti
def get_layout_description_comment(layout, dag):
    physical_qbits = []
    virtual_bit_mapping = layout.get_virtual_bits()
    # one could directly take layout.get_virtual_bits().values(),
    # but that would not necessarily preserve the original ordering
    # of virtual qubits resulting in a wrong layout description
    for qreg in dag.qregs.values():
        for qbit in qreg:
            physical_qbits.append(virtual_bit_mapping[qbit])
    return '// i ' + ' '.join(str(i) for i in physical_qbits)


# author: Elias Foramitti
def get_circuit_cost(qc: QuantumCircuit) -> int:
    instructions = [i[0] for i in qc]
    cost = 0
    for i, inst in enumerate(instructions):
        if inst.name == 'sx' or inst.name == 'x':
            cost += 1
        elif inst.name == 'cx':
            cost += 10
        elif inst.name == 'swap':
            cost += 30
        elif (inst.name != 'rz' and inst.name != 'measure' and inst.name != 'barrier'):
            print(f"{i}th instruction '{inst.name}' not allowed")
    return cost


# check if interacting qubits are connected to each other
# we use a trivial layout because qubits in the transpiled circuit´s gates resemble the physical qubits
# e.g. swap r[2] r[1]; cx r[2] r[0] -> cx-gate´s qubits are listed as ('r', 1), ('r', 0)
# even tough they logically are ('r', 2), ('r', 0)
def check_qubit_connectivity(qc: QuantumCircuit, trivial_layout: Layout, coupling_graph):
    for gate in qc.data:
        for qubit in gate[1]:
            if (False in list(map(lambda g: trivial_layout[qubit] == trivial_layout[g] or
                                            trivial_layout[g] in coupling_graph.neighbors(trivial_layout[qubit]),
                                  gate[1]))) and gate[0].name != 'barrier':
                return False
    return True

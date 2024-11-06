# https://github.com/devsgnr/learning-quantum/blob/1d161e102683c53e28cde0c5520002555409ad38/src/qiskit_03.py
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import NLocal
from qiskit.circuit import ParameterVector
from qiskit.providers.basicaer import QasmSimulatorPy

# Setup Simulator
sim = QasmSimulatorPy()

# Design Rotation Circuit & ParameterVector
rot_param = ParameterVector('r', 2)
rot_circ = QuantumCircuit(2)
rot_circ.rz(rot_param[0], 0)
rot_circ.ry(rot_param[1], 1)

# Design Entanglement Circuit & ParamaterVector
ent_param = ParameterVector('e', 3)
ent_circ = QuantumCircuit(4)
ent_circ.crx(ent_param[0], 0, 1)
ent_circ.crx(ent_param[1], 1, 2)
ent_circ.crx(ent_param[2], 2, 3)

# Setup NLocal Circuit
nlocal_circ = NLocal(
    num_qubits=6,
    rotation_blocks=rot_circ,
    entanglement_blocks=ent_circ,
    entanglement='linear',
    skip_final_rotation_layer=True,
    insert_barriers=True
)

# Draw NLocal Circuit
nlocal_circ.decompose().draw()

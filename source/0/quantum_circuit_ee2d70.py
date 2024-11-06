# https://github.com/Quantum-Creative-Group/quantum_nodes/blob/26234d1b25148306e6a7226e96421838c8ee7583/quantum_nodes/sockets/quantum_circuit.py
from bpy.types import NodeSocket

from qiskit import QuantumCircuit
from animation_nodes.base_types import AnimationNodeSocket


class QuantumCircuitSocket(NodeSocket, AnimationNodeSocket):
    """Quantum circuit socket."""

    bl_idname = "an_QuantumCircuitSocket"
    bl_label = "Quantum Circuit Socket"
    dataType = "Quantum Circuit"
    drawColor = (0.29, 0.0, 0.5, 1.0)
    storable = True
    comparable = False

    @classmethod
    def getDefaultValue(cls):
        return QuantumCircuit(1, 1)

    @classmethod
    def correctValue(cls, value):
        if isinstance(value, QuantumCircuit):
            return value, 0
        return cls.getDefaultValue(), 2

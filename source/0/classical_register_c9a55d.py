# https://github.com/Quantum-Creative-Group/quantum_nodes/blob/26234d1b25148306e6a7226e96421838c8ee7583/quantum_nodes/sockets/classical_register.py
from bpy.types import NodeSocket

from qiskit import ClassicalRegister
from animation_nodes.base_types import AnimationNodeSocket, PythonListSocket


class ClassicalRegisterSocket(NodeSocket, AnimationNodeSocket):
    """Classical register socket."""

    bl_idname = "an_ClassicalRegisterSocket"
    bl_label = "Classical Register Socket"
    dataType = "Classical Register"
    drawColor = (0.73, 0.9, 0.768, 1.0)
    storable = True
    comparable = False

    @classmethod
    def getDefaultValue(cls):
        return ClassicalRegister(1)

    @classmethod
    def correctValue(cls, value):
        if isinstance(value, ClassicalRegister):
            return value, 0
        return cls.getDefaultValue(), 2


class ClassicalRegisterListSocket(NodeSocket, PythonListSocket):
    """List of classical registers socket."""

    bl_idname = "an_ClassicalRegisterListSocket"
    bl_label = "ClassicalRegister List Socket"
    dataType = "ClassicalRegister List"
    baseType = ClassicalRegisterSocket
    drawColor = (0.18, 0.32, 1, 0.5)
    storable = True
    comparable = False

    @classmethod
    def correctValue(cls, value):
        if isinstance(value, list):
            if all(isinstance(element, ClassicalRegister) for element in value):
                return value, 0
        return cls.getDefaultValue(), 2

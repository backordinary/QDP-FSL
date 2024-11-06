# https://github.com/wweronika/qram-grover-search/blob/78d13a07d0782452d804eeec662ce7170b476525/mcz.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates.z import ZGate
from qiskit.circuit.library import MCMT

class MCZ:
    @staticmethod
    def get_mcz(n_bits) -> MCMT:
        z_gate = ZGate()
        mcz = MCMT(z_gate, n_bits - 1,  1, "Multi-controlled Z")
        return mcz

    @staticmethod
    def get_mcxzx(n_bits) -> MCMT: 
        xzx = QuantumCircuit(1)
        xzx.x(0)
        xzx.z(0)
        xzx.x(0)
        xzx_gate = xzx.to_gate()
        mcxzx = MCMT(xzx_gate, n_bits - 1, 1, "Multi-controlled XZX")
        return mcxzx
        
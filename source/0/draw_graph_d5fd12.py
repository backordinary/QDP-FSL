# https://github.com/quantum-compiler/quartz-artifact/blob/119c4e36db0a6c01cae31cc4bb5ca31f20b5d7d1/src/python/utils/draw_graph.py
import sys
from qiskit import QuantumCircuit, transpile


def draw_from_qasm(qasm_file, **kwargs):
    circuit = QuantumCircuit.from_qasm_file(qasm_file)
    print(circuit.draw())
    circuit.draw(circuit, kwargs)


if __name__ == '__main__':
    draw_from_qasm(sys.argv[1], fliename=sys.argv[2])

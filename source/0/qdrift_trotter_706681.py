# https://github.com/Zshan0/trotterization/blob/9b61afab4f0fbd8812084e749436dae9e5b2f92d/src/qdrift_trotter.py
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import QDrift

from parser import parse, get_operator


def qdrift_trotter(operator: PauliEvolutionGate, _r: int) -> QuantumCircuit:
    trotterizor = QDrift(reps=_r)  # order 1 always.
    circ = trotterizor.synthesize(operator)
    return circ


def main():
    string = """2 2 3
    x_0 x_1
    z_0 z_1
    """
    A = parse(string)
    _order, _time, _r = 1, 2.0, 2
    operator = get_operator(A, _time)
    operator = PauliEvolutionGate(operator)
    circ = qdrift_trotter(operator, _r)
    decomposed = circ.decompose()
    print("High-level circuit:")
    print(circ)
    print("Low-level circuit:")
    print(decomposed)


if __name__ == "__main__":
    main()

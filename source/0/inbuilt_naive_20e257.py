# https://github.com/Zshan0/trotterization/blob/9b61afab4f0fbd8812084e749436dae9e5b2f92d/src/inbuilt_naive.py
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter

from parser import parse, get_operator


def inbuilt_trotter(operator: PauliEvolutionGate, _r: int) -> QuantumCircuit:
    trotterizor = LieTrotter(reps=_r)  # order 1 always.
    circ = trotterizor.synthesize(operator)
    return circ


def main():
    string = """3 3 4
    x_0 x_1
    z_0 z_1
    y_2 y_3
    """
    A = parse(string)
    _time, _r = 2.0, 2
    operator = get_operator(A, _time)
    operator = PauliEvolutionGate(operator)
    circ = inbuilt_trotter(operator, _r)
    decomposed = circ.decompose()
    print("High-level circuit:")
    print(circ)
    print("Low-level circuit:")
    print(decomposed)


if __name__ == "__main__":
    main()

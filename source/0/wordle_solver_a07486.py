# https://github.com/jsistos/Quantum-Wordle-Solver/blob/d8f28627bd90f288c45371f0aa8f57e53bee58a3/wordle-solver.py
from string_comparison import *
from custom_wordle import *

from qiskit import Aer
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute
from qiskit.circuit.add_control import add_control
from qiskit.compiler import transpile
from qiskit.extensions import *
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import plot_histogram

if __name__ == "__main__":
    attempts = 6
    curr_attmpt = 0
    newGame = Wordle(attempts, "PHONE")
    while not newGame.won and curr_attmpt < attempts+1:
        user_guess = input("Insert guess:")
        print(newGame.guess_word(user_guess))
        curr_attmpt += 1
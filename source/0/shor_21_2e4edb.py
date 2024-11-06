# https://github.com/awlaplace/Quantum/blob/8b846c8dfac66d08d043565297e8e3089d501190/Shor_factorization/Shor_21.py
## 21の素因数分解をモジュールを用いて実行するためのスクリプト
## 大体1時間ぐらいかかる

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor


def main():
    N = 21
    shor = Shor(N)
    backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=1024)
    result = shor.run(quantum_instance)
    print(f"The list of factors of {N} as computed by the Shor's algorithm is {result['factors'][0]}.")

# https://github.com/evggzzz/Quantum/blob/05253daf8dce9588122bbde94dd4a90a00ef6e7e/qiskit/qiskit_basic_kit.py
#=====モジュールインポート=====
import qiskit
#量子ビットモジュール
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
#古典ビットモジュール
from qiskit import ClassicalRegister
from qiskit import Aer
from qiskit import execute

#=====関数定義=====
# 結果の個数をカウントする
def get_result(qc, shots=1):
    simulator = Aer.get_backend('qasm_simulator')
    return execute(qc, simulator, shots=shots).result().get_counts(qc)
# 行列を表示する
def get_unitary(qc):
    simulator = Aer.get_backend('unitary_simulator') 
    return execute(qc, simulator).result().get_unitary(qc)

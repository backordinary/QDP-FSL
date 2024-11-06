# https://github.com/BensonZhou1991/Circuit-Transformation-via-Monte-Carlo-Tree-Search/blob/d31be045c2baab456f16b93099af550c3a001b5b/post_processing/qiskit_optimization.py
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CXCancellation, Optimize1qGates, CommutativeCancellation
from qiskit.converters import circuit_to_dag

def PostOptimize(cir):
    pm = PassManager()
    pm.append([CXCancellation(), Optimize1qGates(), CommutativeCancellation()])
    out_circ = pm.run(cir)
    return out_circ

if __name__ == '__main__':
    q = QuantumRegister(3)
    cir = QuantumCircuit(q)
    cir.cx(0,1)
    cir.cx(0,1)
    cir.h(1)
    cir.t(1)
    print(cir.draw())
    cir = PostOptimize(cir)
    print(cir.draw())
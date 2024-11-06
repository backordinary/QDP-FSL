# https://github.com/fatginger1024/PQC_Descriptor/blob/ebc57c542eb5448e8e4c0e1ff07637a4a9e4bc5f/tests/test_analyse.py
from pqc import analyse
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def test_analyse():
    Num = 4
    qc = QuantumCircuit(Num)
    x = ParameterVector(r'$\theta$', length=8)
    [qc.h(i) for i in range(Num)]   
    [qc.ry(x[int(2*i)], i) for i in range(Num)]
    [qc.rz(x[int(2*i+1)], i) for i in range(Num)]
    qc.cx(0, range(1, Num))
    circ = qc
    d1,d2 = analyse(circ=qc)
    print(d1,d2)
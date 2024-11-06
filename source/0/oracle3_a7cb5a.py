# https://github.com/UST-QuAntiL/QuantME-UseCases/blob/9403b0a896ad55676416c001539c7589f8efe5fb/2020-ucc/grover/circuits/oracle3.py
from qiskit import QuantumRegister, QuantumCircuit

qc = QuantumCircuit()
q = QuantumRegister(5, 'q')
qc.add_register(q)

# searched bit string: s = 11111
qc.h(q[4])
qc.mct(list(range(4)), 4)
qc.h(q[4])

def get_circuit(**kwargs):
    """Get oracle circuit."""
    return qc
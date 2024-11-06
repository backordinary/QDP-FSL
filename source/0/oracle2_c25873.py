# https://github.com/UST-QuAntiL/QuantME-UseCases/blob/9403b0a896ad55676416c001539c7589f8efe5fb/2020-ucc/bernstein-vazirani/circuits/oracle2.py
from qiskit import QuantumRegister, QuantumCircuit

qc = QuantumCircuit()
q = QuantumRegister(5, 'q')
qc.add_register(q)

# searched bit string: s = 01111 (first bit is ancilla and using qiskit's reverse qubit ordering)
qc.cx(q[0], q[4])
qc.cx(q[1], q[4])
qc.cx(q[2], q[4])
qc.cx(q[3], q[4])

def get_circuit(**kwargs):
    """Get oracle circuit."""
    return qc
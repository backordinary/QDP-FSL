# https://github.com/arshpreetsingh/Qiskit-cert/blob/b2a93d296ee45646bd428570ffa668ea49534398/nand-gate.py
import qiskit
from qiskit import *

def Nand_gate(a, b):
    """
    AND gate
    """
    qc = QuantumCircuit(3, 1)

    # Set up the registers
    if a:
        qc.x(0)
    if b:
        qc.x(1)

    qc.barrier()

    # NAND
    qc.ccx(0, 1, 2)
    qc.x(2) # Here

    qc.barrier()

    # Measure
    qc.measure(2,0)

    print('Depth: {}'.format(qc.depth()))
    job = qiskit.execute(qc, qiskit.BasicAer.get_backend('qasm_simulator'))
    return job.result().get_counts()
res = Nand_gate(False, True)
print(res)

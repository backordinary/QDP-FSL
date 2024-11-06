# https://github.com/Zeff020/QuantumChallenge/blob/476c903b73486062e4b211a65a16c61dcda853c0/Qiskit_Test.py
import qiskit
from qiskit import IBMQ

qr = qiskit.QuantumRegister(1)
cr = qiskit.ClassicalRegister(1)
program = qiskit.QuantumCircuit(qr, cr)

program.measure(qr,cr)

job = qiskit.execute( program, qiskit.BasicAer.get_backend('qasm_simulator') )

print( job.result().get_counts() )

qiskit.IBMQ.load_accounts()
backend = qiskit.providers.ibmq.least_busy(qiskit.IBMQ.backends(simulator=False))
print("We'll use the least busy device:",backend.name())
job = qiskit.execute( program, backend )
print( job.result().get_counts() )

job.result().get_counts()

results = job.result().get_counts()
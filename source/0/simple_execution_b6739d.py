# https://github.com/jwoehr/Quantum-Computing/blob/b31fa04cc32e3a7a0a1c8ad15da1bd73f2ef85b5/platforms/simple_execution.py
# A simple script that sends a quick program to the IBM Q cloud simulator
# An easy test after installing qiskit-terra and qiskit-ibmq-provider
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute

qr = QuantumRegister(2, "userqr")
cr = ClassicalRegister(2, "c0")
qc = QuantumCircuit(qr, cr)
qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.y(qr[0])
qc.x(qr[1])
qc.measure(qr, cr)
provider = IBMQ.load_account()
backend = provider.get_backend("ibmq_qasm_simulator")
job = execute(qc, backend)
result = job.result()
print(result.data())

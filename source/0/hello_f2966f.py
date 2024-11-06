# https://github.com/lukasszz/qiskit-exp/blob/ce14d53735870e7b6ace352629eb4049e9cd6740/hello.py
import qiskit

qr = qiskit.QuantumRegister(1)
cr = qiskit.ClassicalRegister(1)
program = qiskit.QuantumCircuit(qr, cr)
program.measure(qr, cr)
job = qiskit.execute(program, qiskit.Aer.get_backend('qasm_simulator'))
print(job.result().get_counts())

# Frist register token in python console
# from qiskit import IBMQ
# IBMQ.save_account('from: https://quantumexperience.ng.bluemix.net/qx/account/advanced ')


qiskit.IBMQ.load_accounts()
backend = qiskit.backends.ibmq.least_busy(qiskit.IBMQ.backends(simulator=False))
print("We'll use the least busy device:", backend.name())
job = qiskit.execute(program, backend)
print(job.result().get_counts())

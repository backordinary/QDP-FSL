# https://github.com/mi2valley/qiskit-github-actions/blob/c5308eca9d103d4affba2cc3f1bf234adf511279/app/qiskit-ga.py
import os
from qiskit import IBMQ
from qiskit import QuantumCircuit, execute, ClassicalRegister, QuantumRegister
from qiskit.visualization import *
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt


IBMQ_TOKEN = os.getenv("IBMQ_TOKEN")
IBMQ.save_account(str(IBMQ_TOKEN))



# アカウント情報をロードして、使える量子デバイスを確認します
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
provider.backends()

# 最もすいているバックエンドを選びます
large_enough_devices = IBMQ.get_provider(hub='ibm-q', group='open', project='main').backends(
    filters=lambda x: x.configuration().n_qubits > 4 and not x.configuration().simulator and x.configuration().quantum_volume > 16 ) # and not x.configuration().backend_name == "ibmq_manila" and not x.configuration().backend_name == "ibmq_bogota")
print(large_enough_devices)
real_backend = least_busy(large_enough_devices)

print("ベストなバックエンドは " + real_backend.name())

# build quantum circuit
qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(5, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.rz(1.5707963267948966, qreg_q[0])
circuit.sx(qreg_q[0])
circuit.rz(1.5707963267948966, qreg_q[0])
circuit.rz(1.5707963267948966, qreg_q[1])
circuit.sx(qreg_q[1])
circuit.rz(1.5707963267948966, qreg_q[1])
circuit.rz(1.5707963267948966, qreg_q[2])
circuit.sx(qreg_q[2])
circuit.rz(1.5707963267948966, qreg_q[2])
circuit.rz(1.5707963267948966, qreg_q[3])
circuit.sx(qreg_q[3])
circuit.rz(1.5707963267948966, qreg_q[3])
circuit.rz(1.5707963267948966, qreg_q[4])
circuit.sx(qreg_q[4])
circuit.rz(1.5707963267948966, qreg_q[4])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[3])
circuit.measure(qreg_q[4], creg_c[4])

# execute
job = execute(circuit, real_backend)
job_id = job.job_id()
print(job_monitor(job))
job_result = job.result()
job_counts = job_result.get_counts()
print(job_counts)
plot_histogram(job_counts)


# https://github.com/arshpreetsingh/Qiskit-cert/blob/458ec2051820e64695d793207d4bc2435a5e4350/quiz_test_quiz.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, IBMQ, execute
from qiskit.tools.monitor import job_monitor
from read_config import get_api_key

# Connecet with IBM computer.
# Connect with Backend!
IBMQ.enable_account(get_api_key())
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_qasm_simulator')

q = QuantumRegister(1, 'q')
c = ClassicalRegister(1,'c')

qc = QuantumCircuit(q, c)
qc.h(q)
qc.x(q)
qc.x(q)
qc.h(q)
qc.x(q)
qc.measure(q, c)
job = execute(qc, backend, shots=1024)
print(job.result().get_counts(qc))
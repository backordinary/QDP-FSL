# https://github.com/arshpreetsingh/Qiskit-cert/blob/458ec2051820e64695d793207d4bc2435a5e4350/qc_test2.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, IBMQ, execute
from qiskit.tools.monitor import job_monitor
from read_config import get_api_key
import math
import matplotlib.pyplot as plt
# Connecet with IBM computer.
# Connect with Backend!
IBMQ.enable_account(get_api_key())
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_qasm_simulator')


qc = QuantumCircuit(3)
qc.h(0)
qc.z(0)
qc.x(1)
qc.cx(0, 1)
qc.y(1)
qc.x(2)
qc.h(2)
qc.barrier()
Operator
execute()
qc.measure_all()
qft = QuantumCircuit(2)
qft.ry(math.pi / 2, 0)
qft.s(0)
qft.h(1)



new_qc = qc.compose(qft,[1,2])
new_qc.draw(output='mpl')
plt.show()
# https://github.com/Yuval-Toren/t1_t2_qikit/blob/72b8a765fa118241efa51cc1741c39d2e3fe8c75/t1_measure.py
from qiskit import *
import matplotlib as mpl
import qiskit.providers.aer.noise as noise
from qiskit.tools import job_monitor
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# IBMQ.save_account('15ac85484297874b4d7d0c6ebd87428bde35c5952e2b316f12feb1ec985777452fe9c2b4032f64cc956153f5e1b6dac4fe56a5b0fd49f2d5ef732d68d292f410', overwrite=True)
IBMQ.load_account()

results = [0 for _ in range(41)]
t = [k for k in range(20, 421, 10)]
shots_num = 1024
counter = 0
circuits = [QuantumCircuit(1, 1) for _ in range(41)]
for n in range(20, 421, 10):
    circuits[counter].x(0)
    circuits[counter].delay(n, unit='us')
    circuits[counter].measure(0,0)
    counter = counter + 1
counter = 0
provider = IBMQ.get_provider(hub='ibm-q')
device = provider.get_backend('ibmq_lima')
# for n in range(20, 421, 50):

job = execute(transpile(circuits, backend=device, scheduling_method="alap"), backend=device,shots=shots_num)

print(job.job_id())
job_monitor(job)
device_result = job.result()

counts = device_result.get_counts()
results = [100*counts[n]['0']/shots_num for n in range(41)]

plt.title('T1')
plt.xlabel('time between pi pulse and measurement [us]')
plt.ylabel('probability of 0')
plt.plot(t, results)
plt.show()



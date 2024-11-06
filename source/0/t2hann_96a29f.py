# https://github.com/Yuval-Toren/t1_t2_qikit/blob/72b8a765fa118241efa51cc1741c39d2e3fe8c75/t2hann.py
from qiskit import *
from qiskit_experiments.library.characterization.t2hahn import T2Hahn
from qiskit_experiments.test.t2hahn_backend import T2HahnBackend
import matplotlib
from matplotlib import pyplot as plt

IBMQ.load_account()

qubit = 0
conversion_factor = 1e-6 # our delay will be in micro-sec
delays = list(range(0, 50, 1) )
delays = [float(_) * conversion_factor for _ in delays]
number_of_echoes = 1

# Create a T2Hahn experiment. Print the first circuit as an example
exp1 = T2Hahn(qubit=qubit, delays=delays, num_echoes=number_of_echoes)
print(exp1.circuits()[0])

estimated_t2hahn = 20 * conversion_factor
# The behavior of the backend is determined by the following parameters
# backend = T2HahnBackend(
#     t2hahn=[estimated_t2hahn],
#     frequency=[100100],
#     initialization_error=[0.0],
#     readout0to1=[0.02],
#     readout1to0=[0.02],
# )
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_santiago')
exp1.analysis.set_options(p0=None, plot=True)
expdata1 = exp1.run(backend=backend, shots=2000)
expdata1.block_for_results()  # Wait for job/analysis to finish.

# Display the figure
plt.plot(expdata1.data())


for result in expdata1.analysis_results():
    print(result)

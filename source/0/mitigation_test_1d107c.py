# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/mitigation_test.py
from qiskit import *
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, CompleteMeasFitter)

IBMQ.load_account()

nqbit = 3

circuit = QuantumCircuit(nqbit, nqbit)
circuit.h(0)
circuit.cx(0,1)
circuit.cx(1,2)
circuit.measure(range(nqbit), range(nqbit))

simulator = Aer.get_backend('qasm_simulator')
sim_result = execute(circuit, backend= simulator, shots = 1024).result()

plot_histogram(sim_result.get_counts(circuit))


IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
device = provider.get_backend('ibmq_lima')
job = execute(circuit, backend = device, shots = 1024)
print(job.job_id())
job_monitor(job)

device_result = job.result()
plot_histogram(device_result.get_counts(circuit))


cal_circuits,state_labels = complete_meas_cal(qr = circuit.qregs[0], circlabel = 'measerrormitigationcal')

cal_job = execute(cal_circuits, backend = device, shots = 1024, optimization_level = 0)

print(cal_job.job_id())
job_monitor(cal_job)
cal_results = cal_job.result()

plot_histogram(cal_results.get_counts(cal_circuits[3]))


meas_fitter = CompleteMeasFitter(cal_results, state_labels)



meas_filter = meas_fitter.filter
mitigated_result = meas_filter.apply(device_result)

device_counts = device_result.get_counts(circuit)
mitigated_counts = mitigated_result.get_counts(circuit)

plot_histogram([device_counts, mitigated_counts], legend = ['dispositivo , ruido', 'dispositivo, mitigado'])


circuit2 =QuantumCircuit(3,3)
circuit2.x(1)
circuit2.h(0)
circuit2.cx(0,1)
circuit2.cx(1,2)
circuit2.measure([0,1,2],[0,1,2])

sim_counts = execute(circuit2, backend=simulator,shots=1024).result().get_counts(circuit2)
    
device_counts_2 = execute(circuit2, backend= device, shots= 1024).result().get_counts(circuit2)
plot_histogram(device_counts_2)

device_mitigated_counts_2 = meas_filter.apply(device_counts_2)
plot_histogram([device_mitigated_counts_2,sim_counts])
plt.show()
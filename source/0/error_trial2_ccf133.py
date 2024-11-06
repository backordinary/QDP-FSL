# https://github.com/Aarun2/Quantum_Repo/blob/a70505727d2bc83419f470187eb910c8e7a2e4fc/Qiskit_Tutorials/error_trial2.py
from qiskit import *

circuit = QuantumCircuit(3, 3)
circuit.h(0)
circuit.cx(0,1)
circuit.cx(1,2)
circuit.measure([0,1,2],[0,1,2])
circuit.draw()

simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend = simulator, shots = 1024).result()
from qiskit.tools.visualization import plot_histogram
plot_histogram(result.get_counts(circuit))

IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q')
device = provider.get_backend('ibmq_santiago')

job = execute(circuit, backend =device, shots = 1024)
from qiskit.tools.monitor import job_monitor
job_monitor(job)

result = job.result()
plot_histogram(result.get_counts(circuit))

from qiskit.ignis.mitigation.measurement import (complete_meas_cal, CompleteMeasFitter)

cal_circuits, state_labels = complete_meas_cal(qr = circuit.qregs[0], circlabel = 'measerrormitigationcal')

cal_circuits[2].draw()

len(cal_circuits)

cal_job = execute(cal_circuits, 
                 backend = device, 
                 shots = 1024, 
                 optimization_level = 0
                 )
print(cal_job.job_id())
job_monitor(cal_job)
cal_results = cal_job.result()

meas_fitter = CompleteMeasFitter(cal_results, state_labels)
meas_fitter.plot_calibration()
meas_filter = meas_fitter.filter
mitigated_result = meas_filter.apply(device_result)
device_counts = device_result.get_counts(circuit)
mitigated_counts = mitigated_result.get_counts(circuit)
plot_histogram([device_counts, mitigated_counts], legend=['device, noisy', 'device, mitigated'])
circuit2 = QuantumCircuit(3,3)
circuit2.x(1)
circuit2.h(0)
circuit2.cx(0,1)
circuit2.cx(1,2)
circuit2.measure([0,1,2], [0,1,2])
circuit2.draw()

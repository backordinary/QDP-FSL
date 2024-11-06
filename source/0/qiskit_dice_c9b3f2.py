# https://github.com/QuantumQuixxote/QuantumGames/blob/709c2ff2e90bb5abed447bc501cbd27a2810e897/QISKit_Dice.py
from qiskit import IBMQ
IBMQ.save_account('API_KEY')

from qiskit import(QuantumCircuit, execute, Aer)
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, CompleteMeasFitter)
import math
import random

# To run as a simulation, replace all 'backend' with 'simulator', and uncomment:
# simulator = Aer.get_backend('qasm_simulator')

qc = QuantumCircuit(5, 5)
qc.h(0) 
qc.h(1) 
qc.h(2) 
qc.h(3) 
qc.h(4)
qc.measure([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])

IBMQ.load_account()

provider = IBMQ.get_provider('ibm-q')
backend = provider.get_backend('ibmq_ourense')

# If no mitigation is applied, add ', memory=True', and .get_memory() to job.result()

job = execute(qc, backend, shots=4096, optimization_level=0)
results = job.result()

# Mitigation Measurement, will run twice

cal_circuits, state_labels = complete_meas_cal(qr=qc.qregs[0], circlabel='measurement_calibration')
calculatedJob = execute(cal_circuits, backend, shots=4096, optimization_level=0)
calculatedResults = calculatedJob.result()

meas_fitter = CompleteMeasFitter(calculatedResults, state_labels)
meas_filter = meas_fitter.filter
mitigated_result = meas_filter.apply(results)
device_counts = results.get_counts(qc)
mitigated_counts = mitigated_result.get_counts(qc)

output = []

# Reminder, if mitigated results are disabled, add ".get_memory()" to job.result(), and uncomment:

# for x in range(0, 4096):
#     convert = int(mitigated_counts_2[x], 2)
#     output.append(convert)

# And comment out:

for x in mitigated_counts:

    # Since 'mitigated_counts' returns as a dictionary, we need to round how many times 
    # it occurs as a probability, then append the key to the output list by that amount.

    amount = int(round(mitigated_counts[x], 0))
    for y in range(0, amount):
        convert = int(x, 2)
        output.append(convert)

index = math.floor(random.random() * len(output))

maxDiceRoll = 32 
normalizedRoll = (output[index] + 1) / maxDiceRoll
diceSides = 6
result = math.ceil(normalizedRoll * diceSides);
print(result)

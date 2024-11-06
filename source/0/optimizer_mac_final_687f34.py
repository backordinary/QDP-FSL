# https://github.com/KernalPanik/QC_Optimizer/blob/2bc4cfdd3c77a7affd42f61f181b4024fdbe63d6/Optimizer_mac_final.py
import sys
import csv
import os
import qiskit
from Scheduler.Scheduler import get_optimization_type
from Scheduler.Scheduler import AdaptiveScheduler
from Utils import compare_qasm_execs

'''
This module uses runs the optimization using the provided schedule.
'''

print("Running Optimizations based on provided schedule")

optimizations = set()
qasm_file = ""

if(len(sys.argv) < 3):
    print("Not Enough parameters")
    sys.exit(1)
else:
    qasm_file = sys.argv[1]
    with open(sys.argv[2], newline='') as entry:
        reader = csv.reader(entry)
        for row in reader:
            for el in row:
                if(int(el) != 0):
                    optimizations.add(el)

adaptive_scheduler = AdaptiveScheduler()
for opt in optimizations:
    opt_type = get_optimization_type(int(opt))
    if(opt_type != None):
        print(opt_type)
        adaptive_scheduler.add_optimization(opt_type)

optimized_qc = adaptive_scheduler.run_optimization(qiskit.QuantumCircuit.from_qasm_file(qasm_file))

qasm_filename = os.path.splitext(qasm_file)[0]
optimized_qc.qasm(False, qasm_filename+"_optimized.qasm")

verification_succesful = compare_qasm_execs(qasm_file, qasm_filename+"_optimized.qasm")

if(verification_succesful):
    print("Optimized circuit returns same result on simulator.")
else:
    print("Failed to verify that optimized circuit returns same result.")
    sys.exit(1)

sys.exit(0)
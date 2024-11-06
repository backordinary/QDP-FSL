# https://github.com/KernalPanik/QC_Optimizer/blob/e49775ec39526568da6f543d4e1f31d24afcbcd8/Utils.py
'''
This module contains various utilities used in other functions
'''

import csv
from qiskit import Aer, execute
from Scheduler.DagHandler import hash_adj_list
import qiskit
from qiskit.visualization import plot_histogram

def compare_circ_execs(qc1, qc2) -> bool:
    simulator = Aer.get_backend('statevector_simulator')

    job1 = execute(qc1, simulator, shots=1000)
    result1 = job1.result()

    job2 = execute(qc2, simulator, shots=1000)
    result2 = job2.result()

    counts1 = list(result1.get_counts().values())
    counts2 = list(result2.get_counts().values())
    for i in range(0, len(counts1)):
        if(counts1[i] - counts2[i] > 0.000001):
            return False
    return True

def compare_qasm_execs(qasm_file_1, qasm_file_2) -> bool:
    simulator = Aer.get_backend('statevector_simulator')
    qc1 = qiskit.QuantumCircuit.from_qasm_file(qasm_file_1)
    qc2 = qiskit.QuantumCircuit.from_qasm_file(qasm_file_2)
    
    job1 = execute(qc1, simulator, shots=1000)
    result1 = job1.result()

    job2 = execute(qc2, simulator, shots=1000)
    result2 = job2.result()

    counts1 = list(result1.get_counts().values())
    counts2 = list(result2.get_counts().values())
    try:
        fig1 = plot_histogram(result1, color='midnightblue', title="New Histogram")
        fig2 = plot_histogram(result2, color='midnightblue', title="New Histogram")
        
        fig1.savefig(qasm_file_1)
        fig2.savefig(qasm_file_2)
    except ImportError:
        print("Failed to locate libraries to generate histograms..")

    for i in range(0, len(counts1)):
        if(counts1[i] - counts2[i] > 0.000001):
            return False
    
    return True

def put_subdags_into_csv(csv_path: str, subdags: list):
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for subdag in subdags:
            if(len(subdag) < 3):
                for i in range(3 - len(subdag)):
                    subdag.append("i_q0")
            writer.writerow(subdag)

def hash_training_data(training_data_path: str, hashed_data_path: str, col_count = 4) -> str:
    with open(training_data_path, newline='') as entry:
        with open(hashed_data_path, 'w', newline='') as output:
            writer = csv.writer(output, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            reader = csv.reader(entry)

            for row in reader:
                hashed_row = hash_adj_list(row[0:3])

                if(col_count == 4):
                    hashed_row.append(row[-1])
                writer.writerow(hashed_row)
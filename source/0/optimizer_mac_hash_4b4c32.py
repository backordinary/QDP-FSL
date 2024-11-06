# https://github.com/KernalPanik/QC_Optimizer/blob/2bc4cfdd3c77a7affd42f61f181b4024fdbe63d6/Optimizer_mac_hash.py
import qiskit
import sys
import os

from qiskit.dagcircuit import DAGCircuit
from Scheduler.DagHandler import dag_to_list, hash_adj_list, chop_subdag, divide_into_subdags
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from Utils import put_subdags_into_csv, hash_training_data

'''
This module parses the given qasm file into QC, converts it into DAG,
returns a temporary csv file containing hashed circuit data to be analyzed
by Learner class 
'''

qasm_file = ""

if(len(sys.argv) < 2):
    print("Qasm file not provided. Exiting")
    sys.exit(1)
else:
    if(os.path.splitext(sys.argv[1])[1] != ".qasm"):
        print("Incorrect file provided, expected .qasm file")
        sys.exit(1)
    qasm_file = sys.argv[1]

circuit = qiskit.QuantumCircuit.from_qasm_file(qasm_file)
dag = circuit_to_dag(circuit)

adj_list = dag_to_list(dag)
subdags, cx_direction_exists = divide_into_subdags(adj_list)

if(cx_direction_exists):
    pred_file = open("temp_pred.csv", 'a')
    pred_file.write("3,")
    pred_file.close()


for subdag in subdags:
    chopped_dag = list(chop_subdag(subdag))
    put_subdags_into_csv("temp_eval.csv", chopped_dag)

hash_training_data("temp_eval.csv", "temp_eval_hashed.csv", 3)

sys.exit(0)

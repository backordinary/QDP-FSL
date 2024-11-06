# https://github.com/KernalPanik/QC_Optimizer/blob/e49775ec39526568da6f543d4e1f31d24afcbcd8/Optimizer_main.py
import qiskit
import csv
from Utils import compare_qasm_execs, compare_circ_execs
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from Scheduler.DagHandler import dag_to_list, hash_adj_list, chop_subdag, divide_into_subdags
from Utils import put_subdags_into_csv, hash_training_data
from Learner.qmodel import init_test_procedure, init_training_procedure, predict
import os
from Scheduler.Scheduler import get_optimization_type
from Scheduler.Scheduler import AdaptiveScheduler

"""
This module contains a class used to optimize and verify provided 
circuits
"""
class Adaptive_Optimizer():
    def __init__(self):
        pass
    
    def _hash_circuit(self, circuit):
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
    
    def _extract_subdags_from_csv(self, filename):
        subdag_list = list()
        with open(filename, newline='') as entry:
            reader = csv.reader(entry)

            for row in reader:
                floats = [float(item) for item in row]
                subdag_list.append(floats)
        
        return subdag_list

    def _save_predictions_to_csv(self, filename, predictions):
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            pred_set = set(predictions)
            writer.writerow(pred_set)

    def _analyze_circuit(self):
        '''
        Analyze circuit subdags saved in "temp_eval_hashed.csv" file
        using ANN
        '''
        model = init_training_procedure("Learner/training_data.csv", 32)
        init_test_procedure(model, "Learner/training_data.csv", 32)
        subdags = self._extract_subdags_from_csv("temp_eval_hashed.csv")
        pred_names = predict(model, subdags)
        self._save_predictions_to_csv("temp_pred.csv", pred_names)

    def _get_optimizations(self):
        optimizations = list()
        with open("temp_pred.csv", newline='') as entry:
            reader = csv.reader(entry)
            for row in reader:
                for el in row:
                    try:
                        if(int(el) != 0):
                            optimizations.append(el)
                    except ValueError:
                        pass
        return optimizations

    def run_optimization_no_qasm(self, circuit):
        self._hash_circuit(circuit)

        adaptive_scheduler = AdaptiveScheduler()
        self._analyze_circuit()
        optimizations = self._get_optimizations()

        for opt in optimizations:
            opt_type = get_optimization_type((int(opt)))
            if(opt_type != None):
                print(opt_type)
                adaptive_scheduler.add_optimization(opt_type)
        
        optimized_qc = adaptive_scheduler.run_optimization(circuit)
        os.remove("temp_pred.csv")
        os.remove("temp_eval.csv")
        os.remove("temp_eval_hashed.csv")

        verification_succesful = compare_circ_execs(circuit, optimized_qc)

        if(verification_succesful):
            print("Optimized circuit returns same result on simulator.")
        else:
            print("Failed to verify that optimized circuit returns same result.")
        return optimized_qc

    def run_optimization(self, circuit_qasm):
        circuit = qiskit.QuantumCircuit.from_qasm_file(circuit_qasm)
        self._hash_circuit(circuit)

        adaptive_scheduler = AdaptiveScheduler()
        self._analyze_circuit()
        optimizations = self._get_optimizations()

        for opt in optimizations:
            opt_type = get_optimization_type((int(opt)))
            if(opt_type != None):
                print(opt_type)
                adaptive_scheduler.add_optimization(opt_type)
        
        optimized_qc = adaptive_scheduler.run_optimization(circuit)
        qasm_filename = os.path.splitext(circuit_qasm)[0]
        optimized_qc.qasm(False, qasm_filename+"_optimized.qasm")

        verification_succesful = compare_qasm_execs(circuit_qasm, qasm_filename+"_optimized.qasm")

        if(verification_succesful):
            print("Optimized circuit returns same result on simulator.")
        else:
            print("Failed to verify that optimized circuit returns same result.")

        os.remove("temp_pred.csv")
        os.remove("temp_eval.csv")
        os.remove("temp_eval_hashed.csv")
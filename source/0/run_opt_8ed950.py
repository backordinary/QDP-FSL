# https://github.com/hongsy0113/qopt-hsy-/blob/75e9e4bfcc9106b6f353eb2327afd57f7fe39ae2/run_opt.py
from qiskit import QuantumCircuit,transpile,execute,BasicAer
import sys
import subprocess

def print_qc_info (qc):
    print (qc)
    print('The circuit depth : {}'.format (qc.depth()))
    print('The circuit width : {}'.format (qc.width()))
    print('# of qubits : {}'.format (qc.num_qubits))
    print('# of clbits : {}'.format (qc.num_clbits))
    print('# of total gates : {}'.format (qc.size()))
    print('# of nonlocal gates : {}'.format (qc.num_nonlocal_gates()))
    print('gates = ', qc.count_ops())
    print("")

def run_opt (to_analyze):
    filename = "/tmp/qasm1"
    f = open(filename, "w")
    subprocess.call (["./main.native", "-input", to_analyze], stdout=f)
    return filename

def main():
    if(len(sys.argv) > 1):
        filename = str(sys.argv[1])
    else:
        raise FileNotFoundError("\n File name must be specified on argument 1 \n")

    backend = BasicAer.get_backend('qasm_simulator')

    qc_before = QuantumCircuit.from_qasm_file(filename)
    qc_opt3   = transpile(qc_before, backend=backend, seed_transpiler=11, optimization_level=3)
    qc_myopt  = QuantumCircuit.from_qasm_file (run_opt (filename))
   
    print("** Original circuit **") 
    print_qc_info (qc_before)

    print("** After Qiskit optimization **")
    print_qc_info (qc_opt3)

    print("** After my optimization **")
    print_qc_info (qc_myopt)
    
    shots = 1000

    job_before = execute(qc_before, backend, shots=shots)
    job_opt3 = execute(qc_opt3, backend, shots=shots)
    job_myopt  = execute(qc_myopt, backend, shots=shots)

    result_before = job_before.result()
    result_opt3 = job_opt3.result()
    result_myopt = job_myopt.result()

    print("\n -- before optimization --")
    print("The results are: " + str(result_before.get_counts(qc_before)) + "\n")

    print("\n -- after opt3 --")
    print("The results are: " + str(result_opt3.get_counts(qc_opt3)) + "\n")

    print("\n -- after myopt --")
    print("The results are: " + str(result_myopt.get_counts(qc_myopt)) + "\n")

main()

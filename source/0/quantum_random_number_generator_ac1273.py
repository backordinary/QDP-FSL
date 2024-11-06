# https://github.com/ishan616/qiskit-projects-by-ishan/blob/dac9bf25f2ebf1728c691619f34291d14af8b379/quantum_random_number_generator.py
#This code generates a random number between 0-255 using ibm's 1 qubit quantum processor

#import relevant libraries and modules
from qiskit import *
from qiskit.providers.ibmq.job import job_monitor

#load your IBMQ account and get the backends ready
provider = IBMQ.load_account()
qcomp = provider.get_backend("ibmq_armonk") 
M_simulator = Aer.get_backend("qasm_simulator") #local qiskit simulator package as an alternative to ibm's quantum devices

#define the function to generate a random number
def rand():
    qc = QuantumCircuit(1, 1)
    qc.h(0) 
    qc.measure(0, 0)

    #change backend to M_simulator if you don't want to run this on ibm's device

    job = execute(qc, backend=M_simulator, shots=8, memory=True)
    job_monitor(job)
    result = job.result()
    memory = result.get_memory()
    bin = ""

    for a in memory:
        bin += str(a)

    #at this point, bin is an 8-bit binary number which is converted into a decimal number and returned in the nest step

    return int(bin, 2)

print(rand())

# https://github.com/WHLiao/True_Random_Number_Generator_by_IBM_Qiskit/blob/6d1fcbea0fd011b6e2b4ff9fe558e4c73f12e3e4/Random_Number_Qiskit.py
import qiskit
from qiskit import *
from qiskit.tools.monitor import job_monitor
from collections import Counter
IBMQ.save_account('')   # You must fill in a Qiskit API token into IBMQ.save_account('') to run this script.
                        # Please go to IBM Quantum (https://quantum-computing.ibm.com/) to get your own API token and fill it between two single quotations.
                        # If you do not have an IBMid account, you have to create one to get the API token.
                        # The offical Qiskit Youtube account have published a video about how to get the API token can be found in https://youtu.be/M4EkW4VwhcI?t=360 .
                        # The API token part starting from 6:00.
IBMQ.load_account()

qr = QuantumRegister(1)
cr = ClassicalRegister(1)
circuit = QuantumCircuit(qr, cr)
circuit.reset(qr[0])
circuit.h(qr[0])
circuit.measure(qr, cr)
times_of_shots = 15

provider = IBMQ.get_provider('ibm-q')
qcomp = provider.get_backend('ibmq_quito')

def quantum_random_number():

    job = execute(circuit, backend = qcomp, shots = times_of_shots, memory = True)
    job_monitor(job)
    result = job.result()
    Readout = result.get_memory(circuit)

    N = []
    for n in range(times_of_shots):
        N.append(0)

    i = 0
    while i < len(Readout):   
        if (Readout[i] == '0'): 
            N[i] = 0
        elif (Readout[i] == '1'):
            N[i] = (2 ** i) * 1
        else:
            print("Execute Error! The result must be 0 or 1.")
            break
        i += 1

    random_number = sum(N)

    return random_number

QRN = []
j = 0
while j < 2:
    qrn = quantum_random_number()
    QRN.append(qrn)
    print("Excutes time =", (j+1))
    j += 1

QRN.sort()
QRN_counter = dict(Counter(QRN))

file_QRN_50000 = open('file_QRN_50000.txt', 'wt')
k = 0
while k < 0x8000:
    print(k, file = file_QRN_50000, end = ':')
    print(QRN_counter.get(k, 0), file = file_QRN_50000)
    k += 1
file_QRN_50000.close()
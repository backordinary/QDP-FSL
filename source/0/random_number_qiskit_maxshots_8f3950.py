# https://github.com/WHLiao/True_Random_Number_Generator_by_IBM_Qiskit/blob/6d1fcbea0fd011b6e2b4ff9fe558e4c73f12e3e4/Random_Number_Qiskit_Maxshots.py
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
times_of_shots = 8190
RAND_MAX_bit = 15
provider = IBMQ.get_provider('ibm-q')
qcomp = provider.get_backend('ibmq_belem')

def QRN():

    job = execute(circuit, backend = qcomp, shots = times_of_shots, memory = True)
    job_monitor(job)
    result = job.result()
    Readout = result.get_memory(circuit)

    Quantum_Random_Number = []
    q = 0
    while q < times_of_shots:
        
        N = []
        r = q
        while r < (RAND_MAX_bit + q):
            if Readout[r] == '0' :
                N.append(0)
            elif Readout[r] == '1' :
                N.append(2 ** (r - q))
            else:
                print("Execute Error! The result must be 0 or 1.")
                break    
            r += 1

        Quantum_Random_Number.append(sum(N))
        q += RAND_MAX_bit

    return Quantum_Random_Number

Total_QRN = []
s = 0
while s < 100:
    Total_QRN.extend(QRN())
    print("executed job count =", (s+1))
    s += 1

file_total_QRN_raw = open('total_QRN_raw.txt', 'wt')
print(Total_QRN, file = file_total_QRN_raw)
file_total_QRN_raw.close()

Total_QRN.sort()
total_QRN_counter = dict(Counter(Total_QRN))

file_total_QRN_counter = open('total_QRN_counter.txt', 'wt')
t = 0
while t < 0x8000:
    print(t, file = file_total_QRN_counter, end = ':')
    print(total_QRN_counter.get(t, 0), file = file_total_QRN_counter)
    t += 1
file_total_QRN_counter.close()
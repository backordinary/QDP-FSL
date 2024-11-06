# https://github.com/senbar/QuantumComputing/blob/6c28438c98480b823efc912de9a95769c483d6fd/SuperdenseCoding/src/superdense_coding.py
from qiskit import *
from qiskit.visualization import plot_histogram

def create_bell_pair(qc, a, b):
    qc.h(a) # apply h gate
    qc.cx(a,b) # cnot a is control

def encode_message(qc, qubit, msg):
    if msg =="00":
        pass
    elif msg=="10":
        qc.x(qubit)
    elif msg=="01":
        qc.z(qubit)
    elif msg=="11":
        qc.z(qubit)
        qc.x(qubit)

def decode_message(qc, a, b):
    qc.cx(a,b)
    qc.h(a)


qc= QuantumCircuit(2)

create_bell_pair(qc,0,1)
qc.barrier() # todo learn what barrier does, its supposed to be good practice

message="10"
encode_message(qc,0, message)
qc.barrier()


decode_message(qc,0,1)

qc.measure_all()
qc.draw(output="mpl")

backend= Aer.get_backend('qasm_simulator')
job_sim=execute(qc, backend, shots=1024)
sim_result= job_sim.result()

measurement_result=sim_result.get_counts(qc)
print(measurement_result)
plot_histogram(measurement_result)

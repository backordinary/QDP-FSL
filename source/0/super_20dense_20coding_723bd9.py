# https://github.com/NavMohan-24/Quantum-Communication/blob/24becf4ec4586446701ed0e9f99a18c7b6a53a94/Super%20Dense%20Coding.py


from qiskit import*
from qiskit.tools.visualization import*

def create_bellpair(qc,a,b):
    qc.h(a)
    qc.cx(a,b)
    
def encode_message(qc,a,message):
    qc.barrier()
    if message=="00":
        pass
    elif message=="01":
        qc.z(a)
    elif message=="10":
        qc.x(a)
    elif message=="11":
        qc.z(a)
        qc.x(a)
    else:
        print('This message cannot be sent')

def decode_message(qc,a,b):
    qc.barrier()
    qc.cx(a,b)
    qc.h(a) 


qc=QuantumCircuit(2)
create_bellpair(qc,0,1)
message="10"
encode_message(qc,0,message)
decode_message(qc,0,1)
qc.measure_all()
#qc.draw('mpl')


backend=BasicAer.get_backend('qasm_simulator')
measurement_result=execute(qc,backend,shots=1024).result()
counts=measurement_result.get_counts(qc)
plot_histogram(counts)

# Here using a single qubit alice can share 2 bit classical data with bob.
# In similar manner using n qubit we can share 2^n classical bit data.
# There are also a schemes for using a triparitate GHZ entanglement channel for doing superdense coding (arXiv:quant-ph/0610001).
# 
# But this scheme also shares 2 bits of classical info. Here also alice shares only one qubit with bob

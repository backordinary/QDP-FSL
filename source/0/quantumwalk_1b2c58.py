# https://github.com/anuragksv/QuantumLibrary/blob/9e1afd758384335109f231480047632bdf309efe/qulib/QuantumWalk.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute

def QuantumWalk():
    
    #Number of steps
    steps = 2
    
    #Defining the shift gate
    qr = QuantumRegister(3)
    #Circuit for shift operator
    qc = QuantumCircuit (qr, name='shift circuit') 
    
    #Toffoli Gate
    qc.ccx (qr[0], qr[1], qr[2])
    #CNOT Gate
    qc.cx (qr[0], qr[1] )
    qc.x (qr[0])
    qc.x (qr[1])
    qc.ccx (qr[0], qr[1], qr[2])
    qc.x (qr[1])
    qc.cx (qr[0], qr[1])
    
    qc.draw(output='mpl')
    
    #Convert the circuit to Custom Gate
    s_gate = qc.to_instruction() 
    
    q = QuantumRegister (3, name='q')
    c = ClassicalRegister (3, name='c')
    #Primary Circuit
    circuit = QuantumCircuit (q,c)
    
    for i in range(steps):
        
        #Coin Flip
        circuit.h (q[0])
        #Applying Shift 
        circuit.append (s_gate, [q[0],q[1],q[2]])
        
    circuit.measure ([q[0],q[1],q[2]], [c[0],c[1],c[2]])
    circuit.draw(output='mpl')
    
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=1)
    result = job.result()
    print(result.get_counts())  
# https://github.com/anuragksv/QuantumLibrary/blob/9e1afd758384335109f231480047632bdf309efe/build/lib/qulib/SuperdenseCoding.py
#For Circuit Creation, Measurement, and Simulation 
from qiskit import BasicAer, Aer, assemble, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

#Bloch Sphere Visualization
from qiskit.visualization import plot_bloch_multivector, plot_histogram

def SuperdenseCoding():
    epr = QuantumRegister(1, name='epr')
    qubit = QuantumRegister(1, name='qubit')
    c_epr = ClassicalRegister(1, name='c_epr')
    c_qubit = ClassicalRegister(1, name='c_qubit')
    
    qc = QuantumCircuit(epr, qubit, c_epr, c_qubit)
        
    def msg_generator():
        msg_circ = QuantumCircuit(2,2)
        
        msg_circ.h(0)
        msg_circ.h(1)
        msg_circ.measure(0,0)
        msg_circ.measure(1,1)
        
        backend = BasicAer.get_backend('qasm_simulator')
        result = execute(msg_circ, backend, shots=1).result()
    
        for i in result.get_counts().keys():
            return str(i)
        
    msg = msg_generator()
    def epr_pair_generation():
        qc.h(qubit)
        qc.cx(qubit, epr)
        qc.barrier()
    
    def encode():
        if msg[1] == '1':
            qc.x(qubit)
        if msg[0] == '1':
            qc.z(qubit)
        if msg == '00':
            qc.i(qubit)
        qc.barrier()
            
        
    def decode():
        qc.cx(qubit, epr)
        qc.h(qubit)
    
    def simulate():
        qc.measure(qubit, c_qubit)
        qc.measure(epr, c_epr)
        backend = BasicAer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=1).result().get_counts()
        print('Message: {0}\nResult: {1}'.format(msg, result))
        return result
    
    epr_pair_generation()
    encode()
    decode()
    simulate()
    qc.draw(output='mpl')
        
        
# https://github.com/realvindic/Qcrypt/blob/4703ac4b7f54bb10aeaa803127c05db684d3d6ce/randomnum2.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute,IBMQ
from qiskit import Aer

def qauntum():
    q = QuantumRegister(32,'q')
    c = ClassicalRegister(32,'c')
    circuit = QuantumCircuit(q,c)
    circuit.h(q) # Applies hadamard gate to all qubits
    circuit.measure(q,c) # Measures all qubits
    #gets backend
    backend = Aer.get_backend('qasm_simulator');
    #executes backend
    job = execute(circuit, backend, shots=1)
    #gets result
    result = job.result()
    #stores result
    counts = result.get_counts(circuit)
    s = str(counts.int_raw);
    ss = str((s.replace('{','')).replace('}','').rsplit(':',1));
    S = str(str(ss.rsplit(',')).replace('[','').replace(',','').replace("'","").replace('"',"").replace(']',''));
    Ss = S.replace('   ','');
    print(Ss);
    return(Ss);
def write4(skey):
    with open("config.h","a") as f:
        f.write("long double privatekey");
        f.write(" = ");
        f.write(skey);
        f.write(";");
qnum = qauntum();
write4(qnum);

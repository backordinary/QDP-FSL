# https://github.com/anuragksv/QuantumLibrary/blob/9e1afd758384335109f231480047632bdf309efe/build/lib/qulib/GroverSearch.py
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

def ClassicalSearch():
    
    search_list = ['00', '01', '10', '11']
    
    def classical_oracle(ip_element):
        winner = '11'
        if ip_element is winner:
            response = True
        else:
            response = False
        return response
    
    for i,x in enumerate(search_list):
        if classical_oracle(x) is True:
            print('Winner found at index: {0}'.format(i))
            print('Oracle calls: {0}'.format(i+1))
            break
   
    

def GroverSearch():
    
    #define oracle
    oracle = QuantumCircuit(2, name = 'Oracle')
    oracle.cz(0,1)
    oracle.to_gate()
    
    oracle.draw(output = 'mpl')
    
    #Checking StateVectors After Oracle
    sv_back = Aer.get_backend('statevector_simulator')
    grover_part = QuantumCircuit(2,2)
    grover_part.h([0,1])
    grover_part.append(oracle, [0,1])
    
    job_part = execute(grover_part, sv_back)
    result_part = job_part.result()
    
    sv = result_part.get_statevector()
    np.around(sv,2)
    
    print(sv)
    
    #Reflection Operator
    reflection = QuantumCircuit(2, name = 'reflection')
    reflection.h([0,1])
    reflection.z([0,1])
    reflection.cz(0,1)
    reflection.h([0,1])
    reflection.to_gate()
    
    reflection.draw(output = 'mpl')
    
    #Complete Circuit Post Oracle + Reflection
    backend = Aer.get_backend('qasm_simulator')
    grover_circ = QuantumCircuit(2,2)
    grover_circ.h([0,1])
    grover_circ.append(oracle, [0,1])
    grover_circ.append(reflection, [0,1])
    grover_circ.measure([0,1],[0,1])
    
    grover_circ.draw(output = 'mpl')
    
    job = execute(grover_circ, backend, shots=8)
    result = job.result()
    print(result.get_counts())


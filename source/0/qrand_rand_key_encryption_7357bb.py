# https://github.com/itsjackfan/qrke/blob/523aeed560d5639eb40f7bb29a02f3fca07e467b/qrand_rand_key_encryption.py
import numpy as np
import math as m

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor


# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()

### PRELIMINARY FUNCTIONS
def bitcount(counts): 
    return [k for k, v in counts.items() if v == 1][0]

def randint(max):
    n = len(bin(max))-2
    qc = QuantumCircuit(n, n)
    
    for i in range(n):
        qc.h(i)
    
    for i in range(n):
        qc.measure(i, i)
    
    aer_sim = Aer.get_backend('aer_simulator')
    qobj = assemble(qc, aer_sim)
    results = aer_sim.run(qobj, shots=1).result()
    answer = bitcount(results.get_counts())
    
    if int(answer,2) < max and int(answer,2) != 0:
        return int(answer,2)
    else:
        return randint(max)
        
def factor(num):
    backend = Aer.get_backend('aer_simulator')
    quantum_instance = QuantumInstance(backend, shots=1024)
    shor = Shor(quantum_instance=quantum_instance)
    result = shor.factor(num)
    final = result.factors[0]
    
    return final

### KEY AND ENCRYPTION SETTING
# useful bits and bobs
encrypt_dict = {" ":0,"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10,"k":11,"l":12,"m":13,"n":14,"o":15,"p":16,"q":17,"r":18,"s":19,"t":20,"u":21,"v":22,"w":23,"x":24,"y":25,"z":26}
decrypt_dict = {'0':" ",'1':"a",'2':"b",'3':"c",'4':"d",'5':"e",'6':"f",'7':"g",'8':"h",'9':"i",'10':"j",'11':"k",'12':"l",'13':"m",'14':"n",'15':"o",'16':"p",'17':"q",'18':"r",'19':"s",'20':"t",'21':"u",'22':"v",'23':"w",'24':"x",'25':"y",'26':"z"}

# setting p and q
p = int(randint(25))
q = int(randint(25))

# public key
n = p*q
e = 2

# private key
phi = (p-1)*(q-1)
k = 1
if (a % 2 == 1):
    pass
else:
    a // 2

e = factor(a)[0]
print(e)

### ENCRYPT AND DECRYPT FUNCTIONS
def encrypt(message, n, e):
    encrypted_string = ""
    
    for char in range(len(message)):
        encrypted_string += str(encrypt_dict[message[char]])
        
    result = (int(encrypted_string)**e) % n
    return result

def decrypt(encrypted_msg, d, n):
    result = ""
    
    decrypted_num = int((encrypted_msg**d) % n)
    newNum = str(decrypted_num)
    
    for char in range(len(newNum)):
        result += str(decrypt_dict[newNum[char]])
    
    return result
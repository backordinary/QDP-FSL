# https://github.com/thomascherickal/Coding-Portfolio/blob/ed00fadb6f5968edc8aacb8cb3916da54395ebca/Flutter%20UI%20For%20RSA%20Breaker/user_interface_for_rsa_breaker/build/linux/x64/release/bundle/data/flutter_assets/lib/RSA_Breaker-4-bit.py
import enum
from sys import argv
from traceback import print_list
import numpy as np
from qiskit import qiskit
from qiskit import ClassicalRegister
from qiskit import QuantumRegister, QuantumCircuit, execute, Aer, IBMQ        
from math import sqrt,log,gcd
import random
from random import randint
import sys

#### Calculating Modular  Inverse
def mod_inverse(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return -1



#### Checking for primality

def isprime(n):
    if n < 2:
        return False
    elif n == 2:
        return True
    else:
        for i in range(1, int(sqrt(n)) + 1):
            if n % i == 0:
                return False
    return True


#### Generate Keys

def generate_keypair(keysize):
    p = randint(1, 1000)
    q = randint(1, 1000)
    nMin = 1 << (keysize - 1)
    nMax = (1 << keysize) - 1
    primes = [2]
    start = 1 << (keysize // 2 - 1)
    stop = 1 << (keysize // 2 + 1)
    if start >= stop:
        return []
    for i in range(3, stop + 1, 2):
        for p in primes:
            if i % p == 0:
                break
        else:
            primes.append(i)
    q = randint(1, 1000)
    nMin = 1 << (keysize - 1)
    nMax = (1 << keysize) - 1
    primes = [2]
    start = 1 << (keysize // 2 - 1)
    stop = 1 << (keysize // 2 + 1)
    if start >= stop:
        while(primes and primes[0] < start):
            del primes[0]
    # Select two random prime numbers p and q
    while primes:
        p = random.choice(primes)
        primes.remove(p)
        q_values = [q for q in primes if nMin <= p * q <= nMax]
        if q_values:
            q = random.choice(q_values)
            break
    # Calculate n
    n = p * q
    # Calculate phi
    phi = (p - 1) * (q - 1)
    # Select e
    e = random.randrange(1, phi)
    g = gcd(e, phi)
    # Calculate d
    while True:
        e = random.randrange(1, phi)
        g = gcd(e, phi)
        d = mod_inverse(e, phi)
        if g == 1 and e != d:
            break

    return ((e, n), (d, n))




def encrypt(plaintext, package):
    e, n = package
    ciphertext = [pow(ord(c), e, n) for c in plaintext]
    return ''.join(map(lambda x: str(x), ciphertext)), ciphertext




def decrypt(ciphertext, package):
    d, n = package
    plaintext = [chr(pow(c, d, n)) for c in ciphertext]
    return (''.join(plaintext))





## Now lets frame Shor's Algorithm


def period(a,N):
    
    available_qubits = 16 
    r=-1   
    N = 4

    qr = QuantumRegister(available_qubits)   
    cr = ClassicalRegister(available_qubits)
    qc = QuantumCircuit(qr,cr)
    x0 = randint(1, N-1) 
    x_binary = np.zeros(available_qubits, dtype=bool)
    
    for i in range(1, available_qubits + 1):
        bit_state = (N%(2**i)!=0)
        if bit_state:
            N -= 2**(i-1)
        x_binary[available_qubits-i] = bit_state    
    
    for i in range(0,available_qubits):
        if x_binary[available_qubits-i-1]:
            qc.x(qr[i])
    x = x0
    
    while np.logical_or(x != x0, r <= 0):
        r+=1
        qc.measure(qr, cr) 
        qc.x(qr[0])
        qc.x(qr[1])
        qc.x(qr[2])
        qc.cx(qr[2],qr[1])
        qc.cx(qr[1],qr[2])
        qc.cx(qr[2],qr[1])
        qc.cx(qr[1],qr[0])
        qc.cx(qr[0],qr[1])
        qc.cx(qr[1],qr[0])
        qc.cx(qr[3],qr[0])
        qc.cx(qr[0],qr[1])
        qc.cx(qr[1],qr[0])
        
        result = execute(qc,backend = qasm_sim, shots=1024).result()
        counts = result.get_counts()
        
        results = [[],[]]
        for key,value in counts.items(): 
            results[0].append(key)
            results[1].append(int(value))
        s = results[0][np.argmax(np.array(results[1]))]
    return r





def shors_breaker(N):
    N = int(N)
    while True:
        a=randint(0,N-1)
        g=gcd(a,N)
        if g!=1 or N==1:
            return g,N//g
        else:
            r=period(a,N) 
            if r % 2 != 0:
                continue
            elif pow(a,r//2,N)==-1:
                continue
            else:
                p=gcd(pow(a,r//2)+1,N)
                q=gcd(pow(a,r//2)-1,N)
                if p==N or q==N:
                    continue
                return p,q
            
            
def modular_inverse(a,m): 
    a = a % m; 
    for x in range(1, m) : 
        if ((a * x) % m == 1) : 
            return x 
    return 1

class Chooser(enum.Enum): 
    PLAIN   = 1
    ENCRYPT = 2
    DECRYPT = 3
    BREAKER = 4 
    ERROR =  -1

qasm_sim = qiskit.Aer.get_backend('qasm_simulator')


def main():
    
    code = sys.argv[1]
    """_code explained_
    _code = 1: print Plaintext
    _code = 2: print Encrypted Text
    _code = 3: print Decrypted Text
    _code = 4: print result of Shor's Algorithm
    _code = 5: Error    
       _type_: _description_
    """
    bit_length = 4

    public_k, private_k = generate_keypair(2**bit_length)

    #text to encrypt
    plain_txt = sys.argv[2]
    
    #print(plain_txt)
    cipher_txt, cipher_obj = encrypt(plain_txt, public_k)

    #### Encryption
    #print("Encrypted message: {}".format(cipher_txt))

    decrypt_txt = (decrypt(cipher_obj, private_k))

    #### Decryption
    #print("Decrypted message: {}".format(decrypt_txt))
    N_shor = public_k[1]
    p,q = shors_breaker(N_shor)
    phi = (p-1) * (q-1)  
    d_shor = modular_inverse(public_k[0], phi) 

    cracked_text = decrypt(cipher_obj, (d_shor,N_shor))

    # Lets Crack our Cipher Text using Shor's Algorithm
    #print('Message Cracked using Shors Algorithm: {} '.format(cracked_text))    
    
    code = int(argv[1])
    #print(code)   
    if code == Chooser.PLAIN.value: 
        print(plain_txt)
        return plain_txt
    elif code == Chooser.ENCRYPT.value: 
        print(cipher_txt)
        return cipher_txt
    elif code == Chooser.DECRYPT.value: 
        print(decrypt_txt)  
        return decrypt_txt
    elif code == Chooser.BREAKER.value: 
        print(cracked_text)
        return cracked_text
    else: 
        print("Invalid Code")
        return Chooser.ERROR.value  
    

def print(str):
    with open("out.txt", "r") as f:
        data = f.read()
        f.truncate()
        f.close()
    with open("out.txt", "w") as f:
        f.write(str)
        f.close()

main()
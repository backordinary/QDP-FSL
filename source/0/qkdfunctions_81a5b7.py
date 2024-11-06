# https://github.com/fs1132429/QueueMeet/blob/5f6fadd473413b23e520ded4c810331823a0a67a/QKDFunctions.py
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile

# Takes in a bit string msg and basis list
def encode_message(msg, basis):
    enc_msg = []
    n = len(msg)
    for  i in range(n):
        qc = QuantumCircuit(1,1)
        # 0 corresponds to z-basis
        if (basis[i] == 0):
            # apply not gate if current bit is true
            if (msg[i] == 1):
                qc.x(0)
        # 1 corresponds to x-basis
        elif (basis[i] == 1):
            if msg[i] == 0:
                qc.h(0)
            else:
                # apply not gate and then h gate if current bit is true
                qc.x(0)
                qc.h(0)
        # 2 corresponds to y-basis
        elif (basis[i] == 2):
            if msg[i] == 0:
                qc.sdg(0)
            else:
                # Apply sdg gate and then y gate if current bit is true
                qc.sdg(0)
                qc.y(0)
        enc_msg.append(qc)
    return enc_msg

def measure_message(msg, basis, backend=Aer.get_backend('aer_simulator')):
    # Using qiskit backend temporararily
    backend = backend
    mes_msg = []
    n = len(msg)
    for i in range(n):
        # Measure in z-basis
        if basis[i] == 0:
            msg[i].measure(0,0)
        # Measure in x-basis
        elif basis[i] == 1:
            msg[i].h(0)
            msg[i].measure(0,0)
        # Measure in y-basis
        elif basis[i] == 2:
            msg[i].sdg(0)
            msg[i].measure(0,0)
        # Simulate and get measurement
        circ = transpile(msg[i], backend)
        qi_job = backend.run(circ, memory=True)
        result = qi_job.result()
        measured_bit = int(result.get_memory()[0])
        mes_msg.append(measured_bit)
    return mes_msg

# Removes measurement that was in the wrong basis
def remove_bad(a_basis, b_basis, msg):
    gMsg = []
    n = len(a_basis)
    for i in range(n):
        if (a_basis[i] == b_basis[i]):
            gMsg.append(msg[i])
    return gMsg

# Grabs a random sample
def sample_msg(msg, sele):
    sMsg = []
    for i in sele:
        i = np.mod(i, len(msg))
        sMsg.append(msg.pop(i))
    return sMsg

# Compare a random sample
def check_sample(s1, s2, verbose=True):
    if s1 == s2:
        if (verbose):
            print("Keys succesfully exchanged.")
        return True
    else:
        if (verbose):
            print("Someone was listening.")
        return False


def gen_key(size=100, listener=False, verbose=False):
    orig_msg = np.random.randint(2, size=size)
    orig_basis = np.random.randint(3, size=size)
    
    enc_msg = encode_message(orig_msg, orig_basis)
    
    new_basis = np.random.randint(3, size=size)
    if (listener):
        eve_basis = np.random.randint(3, size=size)
        e_msg = measure_message(enc_msg, eve_basis)
        
    mes_msg = measure_message(enc_msg, new_basis)
    
    o_msg = remove_bad(orig_basis, new_basis, orig_msg)
    b_msg = remove_bad(orig_basis, new_basis, mes_msg)
    
    selection = np.random.randint(len(o_msg), size=int(np.sqrt(100)))
    
    s1 = sample_msg(o_msg, selection)
    s2 = sample_msg(b_msg, selection)
    
    if check_sample(s1, s2, verbose):
        return o_msg
    else:
        return 0
    

def gen_keys(agents, listener=False, verbose=False):
    keys = [[] for i in range(agents)]
    for i in range(agents):
        keys[i] = gen_key(listener=listener, verbose=verbose)
        
    return keys

# # Takes in a fname and target person and dictionary of keys
# def wrap_data(fname, target, key_dict):
#     try:
#         # Makes sure the file exists
#         f = open(fname, 'r')
#     except FileNotFoundError:
#         print('File does not exist.')
#         return False
#     # Makes sure the name is in the dict of keys
#     # TODO: seems not quite right, list of keys?? dictionary??
#     if target in key_dict.keys():
#         f.close()
#         return Datapackage(fname, key_dict.get(target))
#     else:
#         return False
#
# # Object to hold fname/file and associated key
# class Datapackage():
#     def __init__(self, fname, pword):
#         self.fname = fname
#         self.pword = pword
#
#     def unlock(self, key):
#         if (key == self.pword):
#             return self.fname
#         else:
#             return False


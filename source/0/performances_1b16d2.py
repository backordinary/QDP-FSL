# https://github.com/StoianAlinBogdan/ProiectLicenta/blob/d15e141ca7b9081f9b6abacacee8cbf87c7c1ede/Performance/Performances.py
from numpy import concatenate
from qiskit import Aer, QuantumCircuit, transpile
from qiskit_finance.circuit.library import UniformDistribution
import math
from timeit import default_timer as Timer
import pandas as pd

'''
@profile-urile sunt pentru lineprofiler
lineprofiler usage:
kernprof -l Performances.py
python -m line_profiler Performances.py.lprof
eventual cu redirect pentru .txt

cprofiler usage:
python -m cProfile -o Performances.prof Performances.py
snakeviz Performances.prof
'''


class QRNG:
    #@profile
    def __init__(self):
        self.QC_Hadamard_1bit = QuantumCircuit(1)
        self.QC_Hadamard_1bit.h(0)
        self.QC_RY_1bit = QuantumCircuit(1)
        self.QC_RY_1bit.ry(math.pi/2, 0)
        self.QC_Hadamard_8bit = QuantumCircuit(8)
        self.QC_Hadamard_8bit.h(range(0, 8))
        self.QC_RY_8bit = QuantumCircuit(8)
        self.QC_RY_8bit.ry(math.pi/2, range(0, 8))
        self.QC_Uniform_1bit = UniformDistribution(1)
        self.QC_Uniform_8bit = UniformDistribution(8)
        self.QRNGs = [self.QC_Hadamard_1bit, self.QC_Hadamard_8bit, self.QC_RY_1bit, self.QC_RY_8bit, self.QC_Uniform_1bit, self.QC_Uniform_8bit]
        for i in range(len(self.QRNGs)):
            self.QRNGs[i].measure_all()
        self.sim = Aer.get_backend('aer_simulator')
        for i in range(len(self.QRNGs)):
            self.QRNGs[i] = transpile(self.QRNGs[i], self.sim, optimization_level=3)

    def concatenate_bits(self, memory):
        numbers = []
        temp = ''
        c = 0
        for i in range(len(memory)):
            temp = temp + memory[i]
            c = c + 1
            if c == 8:
                numbers.append(temp)
                temp = ''
                c = 0
        numbers = [int(x, 2) for x in numbers]
        return numbers

    
    #@profile
    def run_Hadamard_1bit(self):
        qc = self.QRNGs.pop(0)
        result = self.sim.run(qc, shots=800000, memory=True).result()
        memory = result.get_memory()
        numbers = self.concatenate_bits(memory)
        return numbers

    #@profile
    def run_Hadamard_8bit(self):
        qc = self.QRNGs.pop(0)
        result = self.sim.run(qc, shots=100000, memory=True).result()
        memory = result.get_memory()
        for i in range(len(memory)):
            memory[i] = int(memory[i], 2)
        return memory

    def run_RY_1bit(self):
        qc = self.QRNGs.pop(0)
        result = self.sim.run(qc, shots=800000, memory=True).result()
        memory = result.get_memory()
        numbers = self.concatenate_bits(memory)
        return numbers
    
    def run_RY_8bit(self):
        qc = self.QRNGs.pop(0)
        result = self.sim.run(qc, shots=100000, memory=True).result()
        memory = result.get_memory()
        for i in range(len(memory)):
            memory[i] = int(memory[i], 2) 
        return memory    

    def run_Uniform_1bit(self):
        qc = self.QRNGs.pop(0)
        result = self.sim.run(qc, shots=800000, memory=True).result()
        memory = result.get_memory()
        numbers = self.concatenate_bits(memory)
        return numbers

    def run_Uniform_8bit(self):
        qc = self.QRNGs.pop(0)
        result = self.sim.run(qc, shots=100000, memory=True).result()
        memory = result.get_memory()
        for i in range(len(memory)):
            memory[i] = int(memory[i], 2)
        return memory      


class PRNG():
    def __init__(self):
        self.LCG_rand = 6969
        self.LCG_a = 1664525
        self.LCG_c = 1013904223
        self.m = 2 ** 32

        self.jsr = 123456789
        self.jcong = 380116160
        self.z = 362436069
        self.w = 521288629

    def run_LCG(self, shots):
        numbers = []
        for i in range(shots):
            self.LCG_rand =  (self.LCG_a * self.LCG_rand + self.LCG_c) % self.m
        numbers.append(self.LCG_rand)
        return numbers
    
    def run_KISS(self, shots):
        numbers = []
        for i in range(shots):
            #SHR3
            self.jsr = (self.jsr ^ ((self.jsr << 17) % self.m)) % self.m
            self.jsr = (self.jsr ^ ((self.jsr >> 13 ) % self.m)) % self.m
            self.jsr = (self.jsr ^ ((self.jsr << 5) % self.m)) % self.m
            #CONG
            self.jcong = ((69069 * self.jcong) % self.m + 1234567) % self.m
            #MWC
            self.z = ((36969 * (self.z & 65535)) % 2 ** 16 + (self.z >> 16) ) % 2 ** 16
            self.w = ((18000 * (self.w & 65535)) % 2 ** 16 + (self.w >> 16) ) % 2 ** 16
            mwc = ((self.z << 16) + self.w) % self.m
            numbers.append(mwc)
        return numbers
        

        
def brute_force():
    data_for_bruteforce = {
        "Hadamard_1bit": [],
        "Hadamard_8bit": [],
        "RY_1bit": [],
        "RY_8bit": [],
        "Uniform_1bit": [],
        "Uniform_8bit": [],
        "LCG": [],
        "KISS": []
    }

    for i in range(100):
        QRNGs = QRNG()
        prev_time = Timer()
        QRNGs.run_Hadamard_1bit()
        time = Timer()
        data_for_bruteforce['Hadamard_1bit'].append(time - prev_time)
        prev_time = Timer()
        QRNGs.run_Hadamard_8bit()
        time = Timer()
        data_for_bruteforce['Hadamard_8bit'].append(time - prev_time)
        prev_time = Timer()
        QRNGs.run_RY_1bit()
        time = Timer()
        data_for_bruteforce['RY_1bit'].append(time - prev_time)            
        prev_time = Timer()
        QRNGs.run_RY_8bit()
        time = Timer()
        data_for_bruteforce['RY_8bit'].append(time - prev_time)            
        prev_time = Timer()
        QRNGs.run_Uniform_1bit()
        time = Timer()
        data_for_bruteforce['Uniform_1bit'].append(time - prev_time)            
        prev_time = Timer()
        QRNGs.run_Uniform_8bit()
        time = Timer()
        data_for_bruteforce['Uniform_8bit'].append(time - prev_time)            

    PRNGs = PRNG()
    for i in range(100):
        prev_time = Timer()
        PRNGs.run_LCG(100000)
        time = Timer()
        data_for_bruteforce['LCG'].append(time - prev_time)
        prev_time = time
    for i in range(100):
        prev_time = Timer()
        PRNGs.run_KISS(100000)
        time = Timer()
        data_for_bruteforce['KISS'].append(time - prev_time)
        prev_time = time
    
    print(data_for_bruteforce)

    return data_for_bruteforce
    


if __name__ == "__main__":
    data = brute_force()
    df = pd.DataFrame.from_dict(data)
    df.to_csv('data.csv', index=False, header=True)


'''
    # for profiling
    QRNGs = QRNG()
    QRNGs.run_Hadamard_1bit()
    QRNGs.run_Hadamard_8bit()
    QRNGs.run_RY_1bit()
    QRNGs.run_RY_8bit()
    QRNGs.run_Uniform_1bit()
    QRNGs.run_Uniform_8bit()

    PRNGs = PRNG()
    PRNGs.run_LCG(100000)
    PRNGs.run_KISS(100000)
'''
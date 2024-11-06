# https://github.com/StoianAlinBogdan/ProiectLicenta/blob/2a14b7daf47604c315165957c0179a4a1d13af99/Performance/PerformancesNormal.py
# LCG, RY8Bit -> Box-Muller
# vs. Normal_8Bit

from pandas import option_context
from qiskit import Aer, QuantumCircuit, transpile
import math
import numpy as np
from qiskit_finance.circuit.library import NormalDistribution
from timeit import default_timer as Timer
import pandas as pd
import itertools


class RNGs:
    def __init__(self):
        self.LCG_rand = 6969
        self.LCG_a = 1664525
        self.LCG_c = 1013904223
        self.m = 2 ** 32

        self.sim = Aer.get_backend('aer_simulator')
        self.QC_RY_8bit = QuantumCircuit(8)
        self.QC_RY_8bit.ry(math.pi/2, range(0, 8))
        self.QC_RY_8bit.measure_all()

        self.QC_Normal_8bit = NormalDistribution(8)
        self.QC_Normal_8bit.measure_all()
        self.QC_Normal_8bit = transpile(self.QC_Normal_8bit, self.sim, optimization_level=3)


    def run_LCG(self, shots):
        numbers = []
        for i in range(shots):
            self.LCG_rand =  (self.LCG_a * self.LCG_rand + self.LCG_c) % self.m
            numbers.append(self.LCG_rand)
        (z1, z2) = self.Box_Muller(numbers[0:len(numbers)//2], numbers[len(numbers)//2:len(numbers)])
        numbers = list(itertools.chain(*[[x for x in z1], [x for x in z2]]))
        return numbers
    
    def run_RY_8bit(self, shots):
        result = self.sim.run(self.QC_RY_8bit, shots=shots, memory=True).result()
        memory = result.get_memory()
        for i in range(len(memory)):
            memory[i] = int(memory[i], 2) 
        (z1, z2) = self.Box_Muller(memory[0:len(memory)//2], memory[len(memory)//2:len(memory)])
        memory = list(itertools.chain(*[[x for x in z1], [x for x in z2]]))
        return memory  

    def Box_Muller(self, u1, u2):
        u1 = np.array(u1)
        u2 = np.array(u2)
        u1 = u1 / max(u1)
        u2 = u2 / max(u2)
        z1 = np.sqrt(-2 * np.log(u1, where=u1>0)) * np.cos(2 * math.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1, where=u1>0)) * np.sin(2 * math.pi * u2)
        z1 = np.round_((z1 / max(z1))*255)
        z2 = np.round_((z2 / max(z2))*255)
        z1 = np.nan_to_num(z1)
        z2 = np.nan_to_num(z2)
        return (z1.tolist(), z2.tolist())

    def run_Normal_8bit(self, shots):
        result = self.sim.run(self.QC_Normal_8bit, shots=shots, memory=True).result()
        memory = result.get_memory()
        for i in range(len(memory)):
            memory[i] = int(memory[i], 2)
        return memory



if __name__ == "__main__":
    rngs = RNGs()
    data = {
        'LCG': [],
        'RY8Bit': [],
        'Normal8Bit': []
    }
    for i in range(100):
        prev_time = Timer()
        rngs.run_LCG(100000)
        time = Timer()
        data['LCG'].append(time - prev_time)
        print(time - prev_time)
        prev_time = Timer()
        rngs.run_RY_8bit(100000)
        time = Timer()
        data['RY8Bit'].append(time - prev_time)
        print(time - prev_time)
        prev_time = Timer()
        rngs.run_Normal_8bit(100000)
        time = Timer()
        data['Normal8Bit'].append(time - prev_time)
        print(time - prev_time)

    df = pd.DataFrame.from_dict(data)
    df.to_csv('./data_NormalDists.csv', index=False, header=True)

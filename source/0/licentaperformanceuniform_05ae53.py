# https://github.com/StoianAlinBogdan/ProiectLicenta/blob/edafe1623ba614df2cfad21974ca2b864f7e57fd/Performance/LicentaPerformanceUniform.py
from qiskit import Aer, QuantumCircuit, transpile
import math
from qiskit_finance.circuit.library import UniformDistribution
import pandas as pd
import cProfile
import pstats
from timeit import default_timer as Timer

'''
    Tipuri de circuite:
    1 -> hadamard 1 bit
    2 -> ry 1 bit
    3 -> hadamard 8 biti
    4 -> ry 8 biti
    5 -> UniformDistribution
'''

def build_circuit(tip: int) -> QuantumCircuit:
    if tip == 1:
        qc = QuantumCircuit(1)
        qc.h(0)
    elif tip == 2:
        qc = QuantumCircuit(1)
        qc.ry(math.pi/2, 0)
    elif tip == 3:
        qc = QuantumCircuit(8)
        qc.h(range(8))
    elif tip == 4:
        qc = QuantumCircuit(8)
        qc.ry(math.pi, range(8))
    elif tip == 5:
        qc = UniformDistribution(1)
    elif tip == 6:
        qc = UniformDistribution(8)
    qc.measure_all()

    return qc


def run_simulation(qc: QuantumCircuit, tip: int) -> None:
    sim = Aer.get_backend('aer_simulator')
    if tip == 1 or tip == 2 or tip == 5:
        if tip == 5:
            qc = transpile(qc, sim)
        result = sim.run(qc, shots=800000, memory=True).result()
        counts = result.get_counts()
        memory = result.get_memory()
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
        '''
        unique_numbers = list(set(numbers))
        my_counts = {x: numbers.count(x) for x in unique_numbers}
        df1_data = {"number": my_counts.keys(), "counts": my_counts.values()}
        df1 = pd.DataFrame.from_dict(df1_data)
        '''
    else:
        if tip == 6:
            qc = transpile(qc, sim)
        result = sim.run(qc, shots=100000, memory=True).result()
        counts = result.get_counts()
        memory = result.get_memory()


if __name__ == "__main__":
    times = []
    prevtime = Timer()
    qc = build_circuit(1)
    run_simulation(qc, 1)
    time = Timer()
    times.append(time - prevtime)
    prevtime = time
    qc = build_circuit(2)
    run_simulation(qc, 2)
    time = Timer()
    times.append(time - prevtime)
    prevtime = time
    qc = build_circuit(3)
    run_simulation(qc, 3)
    time = Timer()
    times.append(time - prevtime)
    prevtime = time
    qc = build_circuit(4)
    run_simulation(qc, 4)
    time = Timer()
    times.append(time - prevtime)
    prevtime = time
    qc = build_circuit(5)
    run_simulation(qc, 5)
    time = Timer()
    times.append(time - prevtime)
    prevtime = time
    qc = build_circuit(6)
    run_simulation(qc, 6)
    time = Timer()
    times.append(time - prevtime)
    prevtime = time
    print(times)
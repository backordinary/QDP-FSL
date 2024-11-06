# https://github.com/finalwee/CommLab/blob/c8e9d9a718bbc8c73dc13422e52472f8810a9d86/final/main.py
import trevisan_extractor
import universal_hash

import numpy as np
import matplotlib.pyplot as plt
import re
import random

from qiskit import execute
from qiskit import Aer
from qiskit import QuantumRegister,QuantumCircuit 

def plotResult(file):
    data_pattern = re.compile(r"x: .* (\d*)  y: .* (\d*)")
    f = open(file, 'r').readlines()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(0, 2**8), ylim=(0, 2**8))

    x_in = []
    y_in = []
    
    x_out = []
    y_out = []

    count_in = 0

    r = 2**8 - 1

    flag = np.zeros((2**8, 2**8))

    for line in f:
        data = data_pattern.match(line)
        x = int(data.group(1))
        y = int(data.group(2))

        flag[x, y] = flag[x, y] + 1

        if x**2+y**2 < r**2:
            x_in.append(x)
            y_in.append(y)
            count_in = count_in + 1
        else: 
            x_out.append(x)
            y_out.append(y)
    
    repeat = (flag > 1).sum()

    # print(repeat)
    epsilon = repeat / len(f)

    ax.scatter(x_out, y_out, c='r', alpha = 0.5, s = 5)
    ax.scatter(x_in, y_in, c='b', alpha = 0.5, s = 5)
    
    ax.set_title(f'N: 2^{int(np.log2(len(f)))}\n\u03C0: {4*count_in/len(f):.3f}  \u03B5: {epsilon:.3f}', fontsize=18)
    # plt.show()
    plt.savefig(f'./image/trs_qrng__2^{int(np.log2(len(f)))}.png')

def plotResultBit(file):
    f = open(file, 'r').readlines()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(0, 2**8), ylim=(0, 2**8))
    x_in = []
    y_in = []
    
    x_out = []
    y_out = []

    count_in = 0

    r = 2**8 - 1

    for line in f:
        x = int(line[:8],2)
        y = int(line[8:],2)

        if x**2 + y**2 < r**2:
            x_in.append(x)
            y_in.append(y)
            count_in = count_in + 1
        else:
            x_out.append(x)
            y_out.append(y)
    ax.scatter(x_in, y_in, c='b', alpha = 0.5)
    ax.scatter(x_out, y_out, c='r', alpha = 0.5)
    ax.set_title(f'N: 2^{int(np.log2(len(f)))}\nPI: {4*count_in/len(f)}', fontsize=18)
    # plt.savefig(f'./image/qrng_2^{int(np.log2(len(f)))}.png')
    # plt.savefig(f'./image/.png')


def qrng(bit_length, n):
    q=QuantumRegister(bit_length)
    qc=QuantumCircuit(q)
    qc.h(q)
    qc.measure_all()
    
    simulator=Aer.get_backend("qasm_simulator")
    job=execute(qc,simulator,memory=True,shots=n)
    random_bits=job.result().get_memory()
    # random_num=[int(i,2) for i in random_bits]
    # print(random_bits)

    return random_bits

def readFile(file):
    f = open(file, 'r').readlines()

    return f

def testTRSExtractor():
    source_list = readFile('./qrng_source.txt')
    seed_list = readFile('./qrng_seed.txt')
    output_length = 16
    alpha = 0.4
    epsilon = 0.001

    for source in source_list[13478:2**14]:
        seed = seed_list[random.randint(0, 2**8-1)]
        extracted_bit = trevisan_extractor.trsExtractor(seed, source, output_length, alpha, epsilon)
        bit_string = ''.join([str(int(i)) for i in extracted_bit])
        x = bit_string[:8]
        y = bit_string[8:]
        print(f'x: {x} {int(x, 2)}  y: {y} {int(y,2)}')

def testUniversalHash():
    source_list = readFile('./source/32bit_2^16.txt')
    prime = 65537
    m = 2**8

    result = universal_hash.universalHash(source_list, prime, m)

def saveListToTxt(data, file):
    f = open(file, 'a')
    for line in data:
        f.write(line + '\n')

    f.close()

def demo():
    source_bit_length = 333
    source_length = 2**8
    seed_bit_length = 4096
    seed_length = 2**4

    output_bit_length = 16
    alpha = 0.4
    epsilon = 0.01

    flag = np.zeros((2**int(output_bit_length/2), 2**int(output_bit_length/2)))
    count_in = 0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set(xlim=(0, 2**8), ylim=(0, 2**8))

    r = 2**8 - 1

    source_list = qrng(source_bit_length, source_length)
    print("Done generate source.")
    seed_list = qrng(seed_bit_length, seed_length)
    print("Done generate seed.")

    x_list = []
    y_list = []
    c_list = []

    for i,source in enumerate(source_list):
        seed = seed_list[random.randint(0, seed_length - 1)]
        extracted_bit = trevisan_extractor.trsExtractor(seed, source, output_bit_length, alpha, epsilon)
        bit_string = ''.join([str(int(i)) for i in extracted_bit])
        x = bit_string[:8]
        y = bit_string[8:]
        x_int = int(x, 2)
        y_int = int(y, 2)
        print(f'x: {x} {int(x, 2):<4}  y: {y} {int(y,2)}')
        x_list.append(x_int)
        y_list.append(y_int)

        flag[x_int, y_int] = flag[x_int, y_int] + 1

        if x_int**2 + y_int**2 < r**2:
            c = 'b'
            count_in = count_in + 1
        else:
            c = 'r'
        c_list.append(c)
        repeat = (flag > 1).sum()
        error = repeat / (i+1)
        ax.cla()
        ax.scatter(x_list, y_list, s = 5, c = c_list)
        ax.set_title(f'\u03C0: {4*count_in/(i+1):.3f}  \u03B5: {error:.3f}', fontsize=18)

        plt.pause(0.05)
    plt.show()

if __name__ == '__main__':
    # testTRSExtractor()
    # testUniversalHash()

    # plotResult('./result/qrng_result_2^8.txt')
    # plotResult('./result/qrng_result_2^10.txt')
    # plotResult('./result/qrng_result_2^12.txt')
    # plotResult('./result/qrng_result_2^14.txt')

    # plotResult('./result/trs_qrng_2^8.txt')
    # plotResult('./result/trs_qrng_2^10.txt')
    # plotResult('./result/trs_qrng_2^12.txt')
    # plotResult('./result/trs_qrng_2^14.txt')
    # plotResult('./result/trs_qrng_2^16.txt')

    # plotResult('./result/uni_hash_2^8.txt')
    # plotResult('./result/uni_hash_2^10.txt')
    # plotResult('./result/uni_hash_2^12.txt')
    # plotResult('./result/uni_hash_2^14.txt')
    # plotResult('./result/uni_hash_2^16.txt')
    demo()
# https://github.com/Simula-COMPLEX/qusbt/blob/800f794bfaaa10f8d810c755cbe60ea4df641849/code/programs/QRAM.py
import numpy as np
from qiskit import (
    #IBMQ,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
from math import pi
from qiskit.visualization import plot_histogram
from qiskit.tools.visualization import circuit_drawer

def dec2bin(n):
    a = 1
    list = []
    while a > 0:
        a, b = divmod(n, 2)
        list.append(str(b))
        n = a
    s = ""
    for i in range(len(list) - 1, -1, -1):
        s += str(list[i])
    s = s.zfill(9)#input的位数
    return s

def inverse(s):
    s_list = list(s)
    for i in range(len(s_list)):
        if s_list[i] == '0':
            s_list[i] = '1'
        else:
            s_list[i] ='0'
    s = "".join(s_list)
    return s

def QRAM(input,count_times):
    simulator = Aer.get_backend('qasm_simulator')
    qreg = QuantumRegister(4)
    addr = QuantumRegister(1)
    qram0 = QuantumRegister(4)
    qram1 = QuantumRegister(4)
    c = ClassicalRegister(4)

    qc = QuantumCircuit(qreg, addr, qram0, qram1, c)

    input_string = dec2bin(input)
    #print('input:'+str(input_string))
    if input_string[8] == '1':
        qc.x(addr[0])
    if input_string[7-0] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[0])
    if input_string[7-1] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[1])
    if input_string[7-2] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[2])
    if input_string[7-3] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[3])
    if input_string[3-0] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[0])
    if input_string[3-1] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[1])
    if input_string[3-2] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[2])
    if input_string[3-3] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[3])

    qc.barrier()

    qc.h(addr[0])
    qc.p(pi/3,addr[0])
    qc.h(addr[0])
    qc.cswap(addr[0],qram0[0],qram1[0])
    qc.cswap(addr[0],qram0[1],qram1[1])
    qc.cswap(addr[0],qram0[2],qram1[2])
    qc.cswap(addr[0],qram0[3],qram1[3])

    qc.barrier()
    qc.swap(qreg[0],qram0[0])
    qc.swap(qreg[1],qram0[1])
    qc.swap(qreg[2],qram0[2])
    qc.swap(qreg[3],qram0[3])

    qc.barrier()
    control = []
    for j in range(3-0):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-0])
    control = []
    for j in range(3-1):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-1])
    control = []
    for j in range(3-2):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-2])

    qc.barrier()

    qc.measure(qreg,c)


    #circuit_drawer(qc, filename='./QRAM_circuit')

    job = execute(qc,simulator,shots = count_times*100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts

def QRAM_M1(input,count_times):
    simulator = Aer.get_backend('qasm_simulator')
    qreg = QuantumRegister(4)
    addr = QuantumRegister(1)
    qram0 = QuantumRegister(4)
    qram1 = QuantumRegister(4)
    c = ClassicalRegister(4)

    qc = QuantumCircuit(qreg, addr, qram0, qram1, c)

    input_string = dec2bin(input)
    #print('input:'+str(input_string))
    if input_string[8] == '1':
        qc.x(addr[0])
    if input_string[7-0] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[0])
    if input_string[7-1] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[1])
    if input_string[7-2] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[2])
    if input_string[7-3] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[3])
    if input_string[3-0] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[0])
    if input_string[3-1] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[1])
    if input_string[3-2] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[2])
    if input_string[3-3] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[3])

    qc.barrier()

    qc.h(addr[0])
    qc.p(pi/3,addr[0])

    qc.mcp(pi/6,[qram0[0],qram0[1]],addr[0])#M1

    qc.h(addr[0])
    qc.cswap(addr[0],qram0[0],qram1[0])
    qc.cswap(addr[0],qram0[1],qram1[1])
    qc.cswap(addr[0],qram0[2],qram1[2])
    qc.cswap(addr[0],qram0[3],qram1[3])

    qc.barrier()
    qc.swap(qreg[0],qram0[0])
    qc.swap(qreg[1],qram0[1])
    qc.swap(qreg[2],qram0[2])
    qc.swap(qreg[3],qram0[3])

    qc.barrier()
    control = []
    for j in range(3-0):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-0])
    control = []
    for j in range(3-1):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-1])
    control = []
    for j in range(3-2):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-2])

    qc.barrier()

    qc.measure(qreg,c)


    #circuit_drawer(qc, filename='./QRAM_M1_circuit')

    job = execute(qc,simulator,shots = count_times*100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts

def QRAM_M2(input,count_times):
    simulator = Aer.get_backend('qasm_simulator')
    qreg = QuantumRegister(4)
    addr = QuantumRegister(1)
    qram0 = QuantumRegister(4)
    qram1 = QuantumRegister(4)
    c = ClassicalRegister(4)

    qc = QuantumCircuit(qreg, addr, qram0, qram1, c)

    input_string = dec2bin(input)
    #print('input:'+str(input_string))
    if input_string[8] == '1':
        qc.x(addr[0])
    if input_string[7-0] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[0])
    if input_string[7-1] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[1])
    if input_string[7-2] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[2])
    if input_string[7-3] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[3])
    if input_string[3-0] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[0])
    if input_string[3-1] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[1])
    if input_string[3-2] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[2])
    if input_string[3-3] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[3])

    qc.barrier()

    qc.h(addr[0])
    qc.p(pi/3,addr[0])
    qc.h(addr[0])

    qc.ch(qram1[1],addr[0])#M2
    qc.cswap(addr[0],qram0[0],qram1[0])
    qc.cswap(addr[0],qram0[1],qram1[1])
    qc.cswap(addr[0],qram0[2],qram1[2])
    qc.cswap(addr[0],qram0[3],qram1[3])

    qc.barrier()
    qc.swap(qreg[0],qram0[0])
    qc.swap(qreg[1],qram0[1])
    qc.swap(qreg[2],qram0[2])
    qc.swap(qreg[3],qram0[3])

    qc.barrier()
    control = []
    for j in range(3-0):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-0])
    control = []
    for j in range(3-1):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-1])
    control = []
    for j in range(3-2):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-2])

    qc.barrier()

    qc.measure(qreg,c)


    #circuit_drawer(qc, filename='./QRAM_M2_circuit')

    job = execute(qc,simulator,shots = count_times*100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts

def QRAM_M3(input,count_times):
    simulator = Aer.get_backend('qasm_simulator')
    qreg = QuantumRegister(4)
    addr = QuantumRegister(1)
    qram0 = QuantumRegister(4)
    qram1 = QuantumRegister(4)
    c = ClassicalRegister(4)

    qc = QuantumCircuit(qreg, addr, qram0, qram1, c)

    input_string = dec2bin(input)
    #print('input:'+str(input_string))
    if input_string[8] == '1':
        qc.x(addr[0])
    if input_string[7-0] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[0])
    if input_string[7-1] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[1])
    if input_string[7-2] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[2])
    if input_string[7-3] == '1':
        #print('input '+ str(7-i) + '=1')
        qc.x(qram0[3])
    if input_string[3-0] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[0])
    if input_string[3-1] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[1])
    if input_string[3-2] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[2])
    if input_string[3-3] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[3])

    qc.barrier()


    qc.h(addr[0])
    qc.p(pi/3,addr[0])
    qc.h(addr[0])

    qc.cx(qram0[1],addr[0])#M3
    qc.cswap(addr[0],qram0[0],qram1[0])
    qc.cswap(addr[0],qram0[1],qram1[1])
    qc.cswap(addr[0],qram0[2],qram1[2])
    qc.cswap(addr[0],qram0[3],qram1[3])

    qc.barrier()
    qc.swap(qreg[0],qram0[0])
    qc.swap(qreg[1],qram0[1])
    qc.swap(qreg[2],qram0[2])
    qc.swap(qreg[3],qram0[3])

    qc.barrier()
    control = []
    for j in range(3-0):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-0])
    control = []
    for j in range(3-1):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-1])
    control = []
    for j in range(3-2):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-2])

    qc.barrier()

    qc.measure(qreg,c)


    #circuit_drawer(qc, filename='./QRAM_M3_circuit')

    job = execute(qc,simulator,shots = count_times*100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def QRAM_M4(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    qreg = QuantumRegister(4)
    addr = QuantumRegister(1)
    qram0 = QuantumRegister(4)
    qram1 = QuantumRegister(4)
    c = ClassicalRegister(4)

    qc = QuantumCircuit(qreg, addr, qram0, qram1, c)

    input_string = dec2bin(input)
    #print('input:' + str(input_string))
    if input_string[8] == '1':
        qc.x(addr[0])
    if input_string[7 - 0] == '1':
        #print('input ' + str(7 - i) + '=1')
        qc.x(qram0[0])
    if input_string[7 - 1] == '1':
        #print('input ' + str(7 - i) + '=1')
        qc.x(qram0[1])
    if input_string[7 - 2] == '1':
        #print('input ' + str(7 - i) + '=1')
        qc.x(qram0[2])
    if input_string[7 - 3] == '1':
        #print('input ' + str(7 - i) + '=1')
        qc.x(qram0[3])
    if input_string[3 - 0] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[0])
    if input_string[3 - 1] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[1])
    if input_string[3 - 2] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[2])
    if input_string[3 - 3] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[3])

    qc.swap(qram0[1],qram1[0])#M4

    qc.barrier()

    qc.h(addr[0])
    qc.p(pi / 3, addr[0])
    qc.h(addr[0])
    qc.cswap(addr[0], qram0[0], qram1[0])
    qc.cswap(addr[0], qram0[1], qram1[1])
    qc.cswap(addr[0], qram0[2], qram1[2])
    qc.cswap(addr[0], qram0[3], qram1[3])

    qc.barrier()
    qc.swap(qreg[0], qram0[0])
    qc.swap(qreg[1], qram0[1])
    qc.swap(qreg[2], qram0[2])
    qc.swap(qreg[3], qram0[3])

    qc.barrier()
    control = []
    for j in range(3 - 0):
        control.append(qreg[j])
    qc.mcx(control, qreg[3 - 0])
    control = []
    for j in range(3 - 1):
        control.append(qreg[j])
    qc.mcx(control, qreg[3 - 1])
    control = []
    for j in range(3 - 2):
        control.append(qreg[j])
    qc.mcx(control, qreg[3 - 2])

    qc.barrier()

    qc.measure(qreg, c)

    #circuit_drawer(qc, filename='./QRAM_M4_circuit')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts

def QRAM_M5(input, count_times):
    simulator = Aer.get_backend('qasm_simulator')
    qreg = QuantumRegister(4)
    addr = QuantumRegister(1)
    qram0 = QuantumRegister(4)
    qram1 = QuantumRegister(4)
    c = ClassicalRegister(4)

    qc = QuantumCircuit(qreg, addr, qram0, qram1, c)

    input_string = dec2bin(input)
    #print('input:' + str(input_string))
    if input_string[8] == '1':
        qc.x(addr[0])
    if input_string[7 - 0] == '1':
        #print('input ' + str(7 - i) + '=1')
        qc.x(qram0[0])
    if input_string[7 - 1] == '1':
        #print('input ' + str(7 - i) + '=1')
        qc.x(qram0[1])
    if input_string[7 - 2] == '1':
        #print('input ' + str(7 - i) + '=1')
        qc.x(qram0[2])
    if input_string[7 - 3] == '1':
        #print('input ' + str(7 - i) + '=1')
        qc.x(qram0[3])
    if input_string[3 - 0] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[0])
    if input_string[3 - 1] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[1])
    if input_string[3 - 2] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[2])
    if input_string[3 - 3] == '1':
        #print('input ' + str(3 - i) + '=1')
        qc.x(qram1[3])

    qc.barrier()

    qc.h(addr[0])
    qc.p(pi / 3, addr[0])
    qc.h(addr[0])
    qc.cswap(addr[0], qram0[0], qram1[0])
    qc.cswap(addr[0], qram0[1], qram1[1])
    qc.cswap(addr[0], qram0[2], qram1[2])
    qc.cswap(addr[0], qram0[3], qram1[3])

    qc.barrier()
    qc.swap(qreg[0], qram0[0])
    qc.swap(qreg[1], qram0[1])
    qc.swap(qreg[2], qram0[2])
    qc.swap(qreg[3], qram0[3])

    qc.barrier()
    control = []
    for j in range(3 - 0):
        control.append(qreg[j])
    qc.mcx(control, qreg[3 - 0])
    control = []
    for j in range(3 - 1):
        control.append(qreg[j])
    qc.mcx(control, qreg[3 - 1])
    control = []
    for j in range(3 - 2):
        control.append(qreg[j])
    qc.mcx(control, qreg[3 - 2])

    qc.barrier()

    qc.cswap(qreg[3],qreg[1],qreg[0])#M5

    qc.measure(qreg, c)

    #circuit_drawer(qc, filename='./QRAM_M5_circuit')

    job = execute(qc, simulator, shots=count_times * 100)
    result = job.result()
    counts = result.get_counts(qc)

    return counts

def QRAM_specification(input):
    simulator = Aer.get_backend('statevector_simulator')
    qreg = QuantumRegister(4)
    addr = QuantumRegister(1)
    qram0 = QuantumRegister(4)
    qram1 = QuantumRegister(4)
    c = ClassicalRegister(4)

    qc = QuantumCircuit(qreg, addr, qram0, qram1, c)

    input_string = dec2bin(input)
    if input_string[8] == '1':
        qc.x(addr[0])
    if input_string[7-0] == '1':
        qc.x(qram0[0])
    if input_string[7-1] == '1':
        qc.x(qram0[1])
    if input_string[7-2] == '1':
        qc.x(qram0[2])
    if input_string[7-3] == '1':
        qc.x(qram0[3])
    if input_string[3-0] == '1':
        qc.x(qram1[0])
    if input_string[3-1] == '1':
        qc.x(qram1[1])
    if input_string[3-2] == '1':
        qc.x(qram1[2])
    if input_string[3-3] == '1':
        qc.x(qram1[3])

    qc.barrier()

    qc.h(addr[0])
    qc.p(pi/3,addr[0])
    qc.h(addr[0])
    qc.cswap(addr[0],qram0[0],qram1[0])
    qc.cswap(addr[0],qram0[1],qram1[1])
    qc.cswap(addr[0],qram0[2],qram1[2])
    qc.cswap(addr[0],qram0[3],qram1[3])

    qc.barrier()
    qc.swap(qreg[0],qram0[0])
    qc.swap(qreg[1],qram0[1])
    qc.swap(qreg[2],qram0[2])
    qc.swap(qreg[3],qram0[3])

    qc.barrier()
    control = []
    for j in range(3-0):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-0])
    control = []
    for j in range(3-1):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-1])
    control = []
    for j in range(3-2):
        control.append(qreg[j])
    qc.mcx(control, qreg[3-2])

    qc.barrier()

    vector = execute(qc, simulator).result().get_statevector()

    return vector




def probabilityComputing(input):
    pt = []
    t = QRAM_specification(input)
    for i in range(16):
        temp = 0
        for j in range(512):
            temp += abs(t[j*16+i])**2
        pt.append(temp)
    return pt


if __name__ == '__main__':
    print(QRAM(0,2))
    print(QRAM_M1(0, 2))
    print(QRAM_M2(0, 2))
    print(QRAM_M3(0, 2))
    print(QRAM_M4(0, 2))
    print(QRAM_M5(0, 2))
    #print(probabilityComputing(104))



    # a = probabilityComputing(8)
    # for i in range(1024):
    #     print(a[i])
    # print(sum(a))
    # AmplitudeAmplification(3,1)
    # AmplitudeAmplification_M1(3,1)
    # AmplitudeAmplification_M2(3,1)
    # AmplitudeAmplification_M3(3,1)
    # AmplitudeAmplification_M4(3,1)
    # AmplitudeAmplification_M5(3,1)
    # AmplitudeAmplification_M6(3,1)
    # # # f = open('./specification.txt','a')
    # f1 = open('./counts.txt','a')
    # a = probabilityComputing(400)
    # # for i in range(len(a)):
    # #     f.write(str(a[i]))
    # #     f.write('\n')
    # temp = AmplitudeAmplification(400,1)
    # for i in range(len(a)):
    #     i_s = dec2bin(i)
    #     if i_s not in temp:
    #         fre.append(0)
    #     else:
    #         fre.append(temp[i_s])
    #     f1.write(str(fre[i]))
    #     f1.write('\n')

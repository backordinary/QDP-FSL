# https://github.com/yinfupai/LiuYaoForQuantum/blob/b4bf2f409ad5f3fc8e57a31e999c6205453d1595/LiuYao.py
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer
from qiskit.visualization import plot_state_city
from qiskit import execute, IBMQ
import json

backend =  Aer.get_backend('statevector_simulator')

#backend = Aer.get_backend('qasm_simulator')

#归藏出来的为0~7对应乾兑离震巽坎艮坤,对应世爻序列
# gua = GetGua()
# shi = GetShi(gua)

class LiuYao:
    def  __init__(self):
        self.Gua = self.GetGua()
        self.ShiYaoPos = self.GetShi(self.Gua)


        #计算1的个数
    def NumberOfOne(self, n):
        if n < 0:
            n = n & 0xffffffff
        count = 0
        while n:
            count += 1
            n = (n - 1) & n
        return count

    #获取一个爻位
    def GetYao(self):
        #记录出现的反面次数
        yao_count = 0
        circ = QuantumCircuit(1)

        #转换成叠加态
        circ.h(0)
        circ.measure_all()
        job = execute(circ, backend, shots=1)
        result = job.result()
        outputstate = result.get_statevector(circ, decimals=0)

        #如果为0
        if (outputstate[0] == 0):
            yao_count = yao_count + 1
        circ = QuantumCircuit(1)

        #转换成叠加态
        circ.h(0)
        circ.measure_all()
        job = execute(circ, backend, shots=1)
        result = job.result()
        outputstate = result.get_statevector(circ, decimals=0)

        #如果为0
        if (outputstate[0] == 0):
            yao_count = yao_count + 1
        circ = QuantumCircuit(1)

        #转换成叠加态
        circ.h(0)
        circ.measure_all()
        job = execute(circ, backend, shots=1)
        result = job.result()
        outputstate = result.get_statevector(circ, decimals=0)

        #如果为0
        if (outputstate[0] == 0):
            yao_count = yao_count + 1

        #得到一个1时，为少阳，阳变阳
        #得到两个1时，为少阴，阴变阳
        #得到三个1时，为老阳，阳变阴
        #得到零个1时，为老阴，阴变阳

        if (yao_count == 0):
            return [0, 1]
        if (yao_count == 1):
            return [1, 1]
        if (yao_count == 2):
            return [0, 0]
        if (yao_count == 3):
            return [1, 0]


    #获取卦与变卦
    def GetGua(self):
        gua = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
        yao = self.GetYao()
        gua[0][0] = yao[0]
        gua[1][0] = yao[1]
        yao = self.GetYao()
        gua[0][1] = yao[0]
        gua[1][1] = yao[1]
        yao = self.GetYao()
        gua[0][2] = yao[0]
        gua[1][2] = yao[1]
        yao = self.GetYao()
        gua[0][3] = yao[0]
        gua[1][3] = yao[1]
        yao = self.GetYao()
        gua[0][4] = yao[0]
        gua[1][4] = yao[1]
        yao = self.GetYao()
        gua[0][5] = yao[0]
        gua[1][5] = yao[1]
        return gua


    #获取世应
    def GetShi(self,gua):
        sy = [2, 1, 3, 0, 3, 2, 4, 5]
        gz = [0, 0, 0]
        y1 = gua[0][0]
        y2 = gua[0][0 + 3]
        gz[0] = y1 ^ y2
        y1 = gua[0][1]
        y2 = gua[0][1 + 3]
        gz[1] = y1 ^ y2
        y1 = gua[0][2]
        y2 = gua[0][2 + 3]
        gz[2] = y1 ^ y2
        return sy[gz[0] + gz[1] * 2 + gz[2] * 4]

#输出测试结果
if __name__ == "__main__":
    ly = LiuYao()
    gua = ly.Gua
    pos = ly.ShiYaoPos
    if (gua[0][0] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    if (0 == pos):
        print('世', end='\t')
    else:
        print('', end='\t')

    if (gua[1][0] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    print('', end='\r\n')
    if (gua[0][1] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    if (1 == pos):
        print('世', end='\t')
    else:
        print('', end='\t')

    if (gua[1][1] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    print('', end='\r\n')
    if (gua[0][2] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    if (2 == pos):
        print('世', end='\t')
    else:
        print('', end='\t')

    if (gua[1][2] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    print('', end='\r\n')
    if (gua[0][3] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    if (3 == pos):
        print('世', end='\t')
    else:
        print('', end='\t')

    if (gua[1][3] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    print('', end='\r\n')
    if (gua[0][4] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    if (4 == pos):
        print('世', end='\t')
    else:
        print('', end='\t')

    if (gua[1][4] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    print('', end='\r\n')
    if (gua[0][5] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    if (5 == pos):
        print('世', end='\t')
    else:
        print('', end='\t')

    if (gua[1][5] == 0):
        print('- -', end='\t')
    else:
        print('---', end='\t')

    print('', end='\r\n')
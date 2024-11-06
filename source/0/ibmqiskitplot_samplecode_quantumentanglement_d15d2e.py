# https://github.com/QChackSchool/QCTools-IBMQ/blob/cce3a6c3af44a205f92fed9323d77de6e3640bd2/sample%E7%AF%84%E4%BE%8B%E6%AA%94%E6%A1%88/IBMqiskitplot_samplecode_QuantumEntanglement.py
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 04:23:47 2021

@author: NeoChen
"""
from qiskit import IBMQ
import numpy as np
from qiskit import(
  QuantumCircuit,QuantumRegister, ClassicalRegister,
  execute,
  Aer)
from qiskit.visualization import plot_histogram

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Use ibmq_16_melbourne
"""
【python環境請安裝Qiskit】
網址 : https://qiskit.org/documentation/install.html

【IBM帳號註冊方式】
取得token的網址 : https://quantum-computing.ibm.com/login
"""
"""
# 登入帳號使用API
My_token = ''
IBMQ.save_account(My_token,overwrite = True)
IBMQ.load_account()
# 指定要使用的遠端量子電腦設備
# 底下將設備參數設定成跟UI平台一樣就好
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_16_melbourne')

# 遠端的量子電腦用這個(不想用就註解掉)
# simulator = backend
"""
# 模擬的量子電腦(嫌真的太就用這個跑)
simulator = Aer.get_backend('qasm_simulator')


"""
【test code】
#以下可以看成是程式的 hello world
qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)
circuit.h(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
"""

#請把UI產生的Qiskit程式碼貼到下面綠色區域
"""
【circuit Start】
"""
N = 3

qreg_q = QuantumRegister(N, 'q')
creg_c = ClassicalRegister(N, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.h(qreg_q[0])

for i in range(N-1):
    circuit.cx(qreg_q[i],qreg_q[i+1])
    circuit.measure(qreg_q[i], creg_c[i])

circuit.measure(qreg_q[N-1], creg_c[N-1])
"""
【circuit End】
"""

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)

# Draw the circuit
# circuit.draw() + print()
"""
Q : circuit.draw() does not plot anything ?

Ans :
This has just been fixed and will be available in the next release of Qiskit Terra.
In the meantime you can save the figure by doing circuit.draw(output='mpl', filename='my_circuit.png') or you can display the circuit in the terminal by doing print(circuit).
"""
print(circuit.draw())

# Save the circuit
"""
ImportError: The class MatplotlibDrawer needs pylatexenc. to install, run "pip install pylatexenc".

Ans : pip install pylatexenc, and restart your Console.
"""
# 不產生視窗，直接把檔案存成圖片放在程式所處的資料夾
#circuit.draw(output='mpl', filename='my_circuit.jpg')
# 產生視窗，不存檔
#circuit.draw(output='mpl')
circuit.draw(output='text', filename='my_circuit.txt')

# Plot and Show a histogram
# plot_histogram() + plot.show()
"""
Q : plot_histogram doesn't always display ?

Ans : 
This is because Jupyter is displaying the histogram via the returned Figure object from the function instead of being drawn by the ipython matplotlib backend when the function returns (because we close the figure in the function). 
So Spyder is treating the display of the plot like it would display any output from a variable in a cell.
This was a trade-off made for the 0.7 release as all the plotting functions now return a matplotlib figures instead of having no return type.
If we didn't do this everytime you're getting a plot displayed now would print 2 plots. 
I looked at several options and this was the best tradeoff we found.
"""

#plot_histogram(counts)
#plt.pause(0)

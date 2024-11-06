# https://github.com/hflash/profiling/blob/0498dff8c1901591d4428c3149eb3aadf80ac483/circuit_profiling_to_pdf.py
import json
import math
import os
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号
mpl.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

# files = os.listdir('./examples')

# # print(math.ceil(file_num/4))
# output_list = []
# for file in files:
#     circuit = QuantumCircuit.from_qasm_file('./examples/'+file)
#     print(file)
#     num_qubits = len(circuit.qubits)
#     profiling_array = np.zeros((num_qubits, num_qubits))
#     gates_list = []
#     for i in range(len(circuit._data)):
#         item = circuit._data.__getitem__(i)
#         if len(item[1]) == 1:
#             profiling_array[item[1][0].index][item[1][0].index] += 1
#         if len(item[1]) == 2:
#             profiling_array[item[1][0].index][item[1][1].index] += 1
#             gates_list.append([item[1][0].index, item[1][1].index])
#     graph_gates = nx.Graph()
#     graph_gates.add_edges_from(gates_list)
#     connect = nx.algebraic_connectivity(graph_gates, normalized=True, method='lanczos')
#     # files_profiling_array.append(profiling_array)
#     output_dic = {}
#     output_dic['name'] = file
#     output_dic['matrix'] = profiling_array.tolist()
#     output_dic['connectivity'] = connect
#     output_list.append(output_dic)
#
# s = json.dumps(output_list)
# f = open("./profiling_pdf/matrix_v1_simple_complete.json", "w")
# f.write(s)
# f.close()
files = []
files_profiling_array = []
connect = []

ff = open("./profiling_pdf/matrix_v1_simple_complete.json", "r")
ss = ff.read()

content = json.loads(ss)

for item in content:
    files.append(item["name"])
    files_profiling_array.append(np.array(item["matrix"]))
    connect.append(item["connectivity"])


file_num = len(files)
# print(files)
# print(files_profiling_array)
# print(connect)


with PdfPages('./profiling_pdf/profiling_pdf_v1_temp.pdf') as pdf:
    for i in range(math.ceil(file_num/4)):
        fig = plt.figure(figsize=(10, 10))
        if i*4 > file_num:
            break
        title = files[i*4]
        ax1 = fig.add_subplot(221)
        plt.title(title)
        plt.text(2,2,str(connect[i*4]))
        ax1.matshow(files_profiling_array[i*4], cmap='viridis')

        if (i*4 + 1) >= file_num:
            break
        title = files[i * 4 + 1]
        ax2 = fig.add_subplot(222)
        plt.title(title)
        plt.text(2,2,str(connect[i * 4 + 1]))
        ax2.matshow(files_profiling_array[i * 4 + 1], cmap='viridis')

        if (i*4 + 2) >= file_num:
            break
        title = files[i * 4 + 2]
        ax3 = fig.add_subplot(223)
        plt.title(title)
        plt.text(2,2,str(connect[i * 4 + 2]))
        ax3.matshow(files_profiling_array[i * 4 + 2], cmap='viridis')

        if (i*4 + 3) >= file_num:
            break
        title = files[i * 4 + 3]
        ax4 = fig.add_subplot(224)
        plt.title(title)
        plt.text(2,2,str(connect[i * 4 + 3]))
        ax4.matshow(files_profiling_array[i * 4 + 3], cmap='viridis')

        pdf.savefig()
        plt.close()

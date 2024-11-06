# https://github.com/DaisukeIto-ynu/KosakaQ_client/blob/1f995bf547f24132af2dc752a59faf0a43ce9422/test_rb.py
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 12:38:00 2022

@author: daisu
"""

from kosakaq_experiments.KosakaQ_randomized_benchmarking import randomized_benchmarking
from kosakaq_backend import KosakaQBackend 
from qiskit import *
from kosakaq_provider import *


provider = KosakaQProvider("8c8795d3fee73e69271bc8c9fc2e0c12e73e5879")
# print(provider.backends(),"\n")
backend = provider.backends()[0]

# rb = randomized_benchmarking("a",[1, 10, 20, 50, 75, 100, 125, 150, 175, 200],repetition = 1)
# rb = randomized_benchmarking("a",[100],repetition = 1)
rb = randomized_benchmarking(backend,length_vector = [1 ,20, 75, 125, 175], repetition = 5, seed = 5)

# emulator = Aer.get_backend('qasm_simulator')
# life = 10
# while life>0:
#     rb.make_sequence()
#     for i,j in zip(rb.circuits[0],rb.gate_sequence[0]):
#         job = execute( i, emulator, shots=8192 )
#         hist = job.result().get_counts()
#         # print(hist)
#         # print(hist.get("1"))
#         if hist.get("1") is not None:
#             print(j,hist)
#             # print(i)
#             # print(hist)
#     life -= 1
# # for i in rb.gate_sequence[0]:
# #      print(i)

# # 普段使っているのと異なるUser作る

rb.make_sequence()
rb.run()
# print(rb.gate_sequence[29][9])

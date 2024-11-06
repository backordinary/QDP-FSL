# https://github.com/fileaccent/graduate_paper_code/blob/682aabc740d57ec0e6bcece506423443629bcfe5/tool.py
from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit import *
from qiskit.test.mock import FakeMontreal, FakeLagos, FakeGuadalupe
import time

def get_fidelity(circuit, circuit_sim_count, backend): # 计算计数的保真度和后端执行代码的时间
    # circuit 表示量子电路
    # circuit_sim_count 表示模拟结果(即 qwalk/walk.py 中 walk_result 的输出结果)
    # backend 所使用的后端
    # circuit_sim_count = sim_backend.run(circuit, shots=1000).result().get_counts()
    start = time.time()
    circuit_count = backend.run(circuit, shots=1000).result().get_counts()
    print('circuit_count: ', circuit_count)
    # print()
    duration = time.time() - start
    # print('消耗时间: ', duration)
    # print('circuit_sim_count: ', circuit_sim_count)
    # print('circuit_count: ', circuit_count)
    circuit_fidelity = hellinger_fidelity(circuit_sim_count, circuit_count)
    return circuit_fidelity, duration
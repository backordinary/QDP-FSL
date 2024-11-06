# https://github.com/jongjinna/Quantum_IBM/blob/b23bb91311be38d3badf5de8b70dea9ba14507b8/etc/3_e.py
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
# from qiskit import BasicAer
# from qiskit.tools.visualization import plot_state_city

# 2 큐비트 양자 레지스터 생성
q = QuantumRegister(2, 'q')

# 레지스터가 양자회로에서 동작하도록
circuit = QuantumCircuit(q)

# 큐비트 0을 아다마르 게이트 (Gate H)를 사용해 중첩을 만듦
circuit.h(0)

# 큐비트 0을 control하고 큐비트 1을 target하는 CX(Controlled-NOT) 게이트, Bell 상태
circuit.cx(0, 1)

# 2비트의 고전 레지스터 생성
c = ClassicalRegister(2, 'c')

# 양자+고전 회로 설정 베리어 생성
meas = QuantumCircuit(q, c)
meas.barrier(q)

# 각 큐비트를 측정
meas.measure(q, c)

# 고전적 비트에 큐비트를 측정한 정보가 나오도록 더하기 연산 조작
# 고전적 비트는 0으로 초기화된 상태에서 큐비트의 측정값을 덧셈하여 값을 가져옴
qc = meas + circuit

print(qc.draw())

circuit = QuantumCircuit(2, 2)

qc = meas + circuit

print(qc.draw())


# Note : https://m.blog.naver.com/simhc0714/221570451068

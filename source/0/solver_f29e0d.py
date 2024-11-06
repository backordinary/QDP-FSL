# https://github.com/PVNkT/HHL/blob/29c91c32d4e251e99aeb7d7b43ce27b197d7e7e1/solver.py
import numpy as np
from circuit import circuit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.algorithms.linear_solvers.hhl import HHL
from qiskit.quantum_info import Statevector
from normalize import calculate_vector
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from value_setter import value_setter
from qiskit.opflow import (
    Z,
    I,
    StateFn,
    TensoredOp)
from qiskit.tools import job_monitor

def HHL_my(A, b, wrap = True, measurement = None):
    """
    measurement == "statevector": use statevector simulation
    measurement == "norm": use observable to calculate Euclidian norm
    measurement == "real_{backend name}": use real quantum computer
    else: use noraml measurement
    """
    #기초 회로 구성, 입력값: A(행렬), b(벡터), nl(사용하는 qubit의 수, 높을 수록 정확도가 높아짐), delta(evolution time t와 reciprocal 과정에서의 scaling을 결정), wrap(회로를 합쳐서 볼지 결정)
    nl, evolution_time, delta, neg_vals = value_setter(A)
    qc = circuit(A, b, nl, evolution_time, delta, neg_vals, wrap = wrap, measurement = measurement)
    eigen_min = min(np.abs(np.linalg.eigvals(A)))
    nb = int(np.log2(len(b)))
    if measurement == "statevector":
        naive_sv = Statevector(qc).data
        #qubit수를 확인
        num_qubit = qc.num_qubits
        #상태 벡터에서 필요한 상태만을 골라서 저장함
        #print(naive_sv)
        naive_full_vector = np.array([naive_sv[2**(num_qubit-1)+i] for i in range(len(b))])
        #실수 부분만 취함
        naive_full_vector = np.real(naive_full_vector)
        #얻어진 벡터를 normalize하여 반환
        #print(naive_full_vector)
        vector = naive_full_vector/eigen_min

    elif measurement == "norm":
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2
        observable = one_op ^ TensoredOp((nl) * [zero_op]) ^ (I ^ nb)
        norm = (~StateFn(observable) @ StateFn(qc)).eval()
        vector = np.real(np.sqrt(norm)/eigen_min)

    elif measurement[:4] == "real":
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q-skku', group='hanyang-uni', project='hu-students')
        backend = provider.get_backend(measurement[5:])
        
        #backend = Aer.get_backend('aer_simulator')
        shots = 8192
        t_qpe = transpile(qc, backend)
        qobj = assemble(t_qpe, shots=shots)
        job = backend.run(qobj)
        job_monitor(job)
        results = job.result()
        
        #시뮬레이션된 결과를 dictionary로 받음
        answer = results.get_counts()
        #실험 결과를 통해서 noramlize된 결과를 얻음
        vector = calculate_vector(answer, int(np.log2(len(b))))
    else:
        #양자 회로를 시뮬레이션하는 코드
        
        backend = Aer.get_backend('aer_simulator')
        shots = 8192
        t_qpe = transpile(qc, backend)
        qobj = assemble(t_qpe, shots=shots)
        job = backend.run(qobj)
        results = job.result()
        
        #시뮬레이션된 결과를 dictionary로 받음
        answer = results.get_counts()
        #실험 결과를 통해서 noramlize된 결과를 얻음
        vector = calculate_vector(answer, int(np.log2(len(b))))
    return vector


def HHL_qiskit(A,b, measurement = None):
    #backend 설정
    backend = Aer.get_backend('aer_simulator')
    #qiskit HHL 코드를 불러옴
    hhl = HHL(quantum_instance=backend)
    #A, b에 대해서 HHL 회로를 구성
    solution = hhl.solve(A, b)
    #만들어진 회로를 그림으로 저장
    #print(solution.euclidean_norm)
    solution.state.draw("mpl").savefig("image/HHL_circuit_qiskit.png")
    #연산된 상태를 상태 벡터의 형태로 결과를 얻음
    naive_sv = Statevector(solution.state).data
    #qubit수를 확인
    num_qubit = solution.state.num_qubits
    #상태 벡터에서 필요한 상태만을 골라서 저장함
    naive_full_vector = np.array([naive_sv[2**(num_qubit-1)+i] for i in range(len(b))])
    #실수 부분만 취함
    naive_full_vector = np.real(naive_full_vector)
    #얻어진 벡터를 normalize하여 반환
    if measurement == "norm":
        return np.linalg.norm(naive_full_vector)/min(np.abs(np.linalg.eigvals(A)))
    else:
        return naive_full_vector/np.linalg.norm(naive_full_vector)
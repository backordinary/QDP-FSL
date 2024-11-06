# https://github.com/PankidT/qwantaBSC/blob/cd45b908a7336f8a03706142057601a20335399c/qwanta-main/Tomography/Tomography_physical.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from copy import deepcopy
import qiskit.quantum_info as qi
import ast
import numpy as np

# Generate result from simulation
# 0:I, 1:XZ, 2:X, 3:Z
def Tomography(data, measurement_error=0, basis_measure_num=1000):
    measurement_basis = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    x = np.random.random((int(len(measurement_basis*basis_measure_num)), 2))
    all_results = []
    for index, basis in enumerate(measurement_basis):
        measurement_outcome = {}
        for operator in range(basis_measure_num):
            operator_index = index*basis_measure_num + operator
            qr = QuantumRegister(2)
            cr = ClassicalRegister(2)
            qc = QuantumCircuit(qr, cr)

            qc.h(qr[0])
            qc.cx(qr[0], qr[1])

            # apply simulation result
            if data['qubit1'][operator_index] == 1:
                qc.x(qr[0])
                qc.z(qr[0])
            elif data['qubit1'][operator_index] == 2:
                qc.x(qr[0])
            elif data['qubit1'][operator_index] == 3:
                qc.z(qr[0])

            if data['qubit2'][operator_index] == 1:
                qc.x(qr[1])
                qc.z(qr[1])
            elif data['qubit2'][operator_index] == 2:
                qc.x(qr[1])
            elif data['qubit2'][operator_index] == 3:
                qc.z(qr[1])

            # measurement basis 
            for i, j in enumerate(basis):
                if j == 'X':
                    qc.h(qr[i])
                    # apply measurement error
                    if x[operator_index][i] < measurement_error:
                        qc.x(qr[i])
                elif j == 'Y':
                    qc.sdg(qr[i])
                    qc.h(qr[i])
                    # apply measurement error
                    if x[operator_index][i] < measurement_error:
                        qc.x(qr[i])
                elif j == 'Z':
                    # apply measurement error
                    if x[operator_index][i] < measurement_error:
                        qc.x(qr[i])
                else:
                    raise ValueError('Damm')
            
            qc.measure(qr, cr)
            result = execute(qc, Aer.get_backend('qasm_simulator'), shots=1).result().get_counts()
            for res in result:
                if res in measurement_outcome:
                    measurement_outcome[res] += 1
                else:
                    measurement_outcome[res] = 1
        all_results.append(measurement_outcome)

    # Bell pair circuit

    qr_bell = QuantumRegister(2)
    qc_bell = QuantumCircuit(qr_bell)
    qc_bell.h(qr_bell[0])
    qc_bell.cx(qr_bell[0], qr_bell[1])
    qc_bell.barrier()

    target_state_bell = qi.Statevector.from_instruction(qc_bell)

    qst_qc = state_tomography_circuits(qc_bell, [qr_bell[0],qr_bell[1]])
    #Run in Aer
    job = execute(qst_qc, Aer.get_backend('qasm_simulator'), shots=basis_measure_num)
    raw_results = job.result()

    new_result = deepcopy(raw_results)

    for resultidx, _ in enumerate(raw_results.results):
        new_result.results[resultidx].data.counts = all_results[resultidx]

    tomo_bell = StateTomographyFitter(new_result, qst_qc)
    # Perform the tomography fit
    # which outputs a density matrix
    rho_fit_bell = tomo_bell.fit(method='lstsq')

    F_bell = qi.state_fidelity(rho_fit_bell, target_state_bell)
    return F_bell

def AddMeasurementError(data, measurement_error, basis_measure_num=1000):
    measurement_basis = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    x = np.random.random((9000, 2))
    for index, basis in enumerate(measurement_basis):
        for operator in range(basis_measure_num):
            operator_index = index*basis_measure_num + operator
            for ii, base in enumerate(basis):
                if x[operator_index][ii] < measurement_error:
                    if base in ['X', 'Y']:
                        if data['qubit1'][operator_index] == 0:
                            data['qubit1'][operator_index] = 3
                        elif data['qubit1'][operator_index] == 1:
                            data['qubit1'][operator_index] = 2
                        elif data['qubit1'][operator_index] == 2:
                            data['qubit1'][operator_index] = 1
                        else:
                            data['qubit1'][operator_index] = 0
                    else:
                        if data['qubit1'][operator_index] == 0:
                            data['qubit1'][operator_index] = 2
                        elif data['qubit1'][operator_index] == 1:
                            data['qubit1'][operator_index] = 3
                        elif data['qubit1'][operator_index] == 2:
                            data['qubit1'][operator_index] = 0
                        else:
                            data['qubit1'][operator_index] = 1
    return data
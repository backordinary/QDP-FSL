# https://github.com/heineaabo/QuantumAlgorithms/blob/9ef038b5b2b721efada8360e76382b055a1fd38b/quantum_algorithms/attributes/mitigation.py
from qiskit import QuantumRegister,execute
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter


def get_measurement_fitter(l,backend,device,shots):
    qb = QuantumRegister(l)
    qubit_list = list(range(l))
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list,qr=qb,circlabel='mcal')

    # Execute calibration circuits 
    job = execute(meas_calibs,
                  backend=backend,
                  shots=shots,
                  noise_model=device.noise_model,
                  coupling_map=device.coupling_map,
                  basis_gates=device.basis_gates)
    cal_results = job.result()

    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    #meas_filter = meas_fitter.filter
    #mitigated_results = meas_filter.apply(results)
    #mitigated_counts = mitigated_results.get_counts(0)
    return meas_fitter



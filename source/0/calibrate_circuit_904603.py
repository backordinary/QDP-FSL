# https://github.com/akashdhruv/FlowX/blob/a4ad14b57736cb5b58dc9c89a067c0adc5d3b5c3/flowx/quantum/_interface/_calibrate_circuit.py
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit import IBMQ, Aer, BasicAer, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer.noise import NoiseModel


def calibrate_circuit_QASM(register, backend, calibrate):

    provider = IBMQ.load_account()
    device_backend = provider.get_backend(backend)

    device = provider.get_backend(
        "ibmq_qasm_simulator"
    )  # Aer.get_backend('qasm_simulator') #
    print("Running on device: ", device)

    meas_fitter, noise_model, basis_gates = [None] * 3

    if calibrate:
        noise_model = NoiseModel.from_backend(device_backend)
        basis_gates = noise_model.basis_gates
        circuit, state_labels = complete_meas_cal(
            qubit_list=list(range(0, len(register))), qr=register, circlabel="mcal"
        )

        job = execute(
            circuit,
            backend=device,
            shots=1024,
            noise_model=noise_model,
            basis_gates=basis_gates,
        )
        job_monitor(job, interval=2)
        cal_results = job.result()

        meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel="mcal")

    return meas_fitter, device, [noise_model, basis_gates]


def calibrate_circuit_IBMQ(register, backend, calibrate):

    provider = IBMQ.load_account()
    device_backend = provider.get_backend(backend)

    device = device_backend
    print("Running on device: ", device)

    meas_fitter, noise_model, basis_gates = [None] * 3

    if calibrate:
        circuit, state_labels = complete_meas_cal(
            qubit_list=list(range(0, len(register))), qr=register, circlabel="mcal"
        )

        job = execute(circuit, backend=device, shots=1024, max_credits=10)
        job_monitor(job, interval=2)

        cal_results = job.result()

        meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel="mcal")

    return meas_fitter, device, [noise_model, basis_gates]

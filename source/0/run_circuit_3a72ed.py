# https://github.com/akashdhruv/FlowX/blob/a4ad14b57736cb5b58dc9c89a067c0adc5d3b5c3/flowx/quantum/_interface/_run_circuit.py
from qiskit import IBMQ, Aer, BasicAer, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer.noise import NoiseModel


def run_circuit_QASM(
    device, noise, fitter, circuit, quantum_register, classical_register
):

    circuit.measure(quantum_register, classical_register)

    job = execute(
        circuit, backend=device, shots=1024, noise_model=noise[0], basis_gates=noise[1]
    )
    job_monitor(job, interval=2)

    noisy_results = job.result()
    noisy_answer = noisy_results.get_counts()

    mitigated_results, mitigated_answer = [noisy_results, noisy_answer]

    if fitter:
        mitigated_results = fitter.filter.apply(noisy_results)
        mitigated_answer = mitigated_results.get_counts()

    results = {"noisy": noisy_results, "mitigated": mitigated_results}
    answer = {"noisy": noisy_answer, "mitigated": mitigated_answer}

    return results, answer


def run_circuit_IBMQ(
    device, noise, fitter, circuit, quantum_register, classical_register
):

    circuit.measure(quantum_register, classical_register)

    job = execute(circuit, backend=device, shots=1024, max_credits=10)
    job_monitor(job, interval=2)

    noisy_results = job.result()
    noisy_answer = noisy_results.get_counts()

    mitigated_results, mitigated_answer = [noisy_results, noisy_answer]

    if fitter:
        mitigated_results = fitter.filter.apply(noisy_results)
        mitigated_answer = mitigated_results.get_counts()

    results = {"noisy": noisy_results, "mitigated": mitigated_results}
    answer = {"noisy": noisy_answer, "mitigated": mitigated_answer}

    return results, answer

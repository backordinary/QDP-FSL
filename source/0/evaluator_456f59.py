# https://github.com/jeffreyjgong/cpsc447-project/blob/59c923738d21124efdd3575881b378b33eca0ded/src/evaluator.py
from starter import MY_TROTTER_STEPS, my_trotter  # feel free to replace to your module

"""
Recommended development procedure:
1. use noiseless simulator to verify your trotterization is correct: as `trotter_steps` goes high, fidelity should approach 100%
2. use jakarta simulator to see how it performs on noisy simulator, observe what's the bottleneck
3. use real jakarta to test fidelity. Since the hardware resource is limited, you may wait for a long time before it executes
"""
USE_NOISELESS_SIMULATOR = True
USE_REAL_JAKARTA = False
if USE_NOISELESS_SIMULATOR:
    print("[backend] noiseless simulator")
    print("\033[93m[warning] this result is not for grading\033[0m")
else:
    from qiskit import IBMQ
    assert False, "please fill in your token from IBM-Q"
    # IBMQ.save_account(TOKEN)  # replace TOKEN with your API token string (https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq)
    IBMQ.load_account()
    if USE_REAL_JAKARTA:
        print("[backend] jakarta")
    else:
        print("[backend] jakarta simulator")

"""
Here we provide you some useful evaluation functions
"""
def main():
    """ evaluate fidelity for a single case, [warning] this is NOT your grading fidelity """
    # print(evaluate_fidelity(MY_TROTTER_STEPS))

    """ This will scan over some range of trotter_steps, for you to find the optimal value of trotter_steps """
    # scan_trotter_steps(4, 13)

    """
    [grading fidelity]
    This is what we do when grading: we'll not only test the final state, but the state after every trotter step
    If you do trotterization correctly, the fidelity should go lower with more trotter steps (but can have fluctuation if your fidelity is high): noise accumulates
    If you observe an extreme low fidelity in the middle, you may need to debug whether your trotterization is correct (basically we want to prevent a solution that simply recovers the final state)
    """
    do_grading(MY_TROTTER_STEPS)


def scan_trotter_steps(start, end, step=1):
    fidelities = []
    for trotter_steps in range(start, end, step):
        fidelity, fidelity_stderr = evaluate_fidelity(trotter_steps)
        print(f"trotter_steps {trotter_steps}: fidelity = {fidelity} stderr = {fidelity_stderr}")
        fidelities.append((trotter_steps, fidelity, fidelity_stderr))
    return fidelities

def do_grading(trotter_steps):
    # evaluate every fidelity after each trotter step
    fidelities = []
    for intermediate_trottor in range(1, trotter_steps+1):
        fidelity, fidelity_stderr = evaluate_fidelity(trotter_steps, intermediate_trottor)
        print(f"intermediate_trottor {intermediate_trottor}: fidelity = {fidelity} stderr = {fidelity_stderr}")
        fidelities.append((intermediate_trottor, fidelity, fidelity_stderr))
    # find the minimum fidelity in every case
    min_fidelity = fidelities[0][0]
    for intermediate_trottor, fidelity, fidelity_stderr in fidelities:
        if fidelity < min_fidelity:
            min_fidelity = fidelity
    print(f"[grade] fidelity = {min_fidelity}")
    return min_fidelity, fidelities







"""
Library code
"""

import numpy as np
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, execute, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter
# Import state tomography modules
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity
from qiskit.opflow import Zero, One, I, X, Y, Z

backend = QasmSimulator()

if not USE_NOISELESS_SIMULATOR:
    provider = IBMQ.get_provider(hub='ibm-q-education', group='yale-uni-2', project='cpsc647-quantum')
    jakarta = provider.get_backend('ibmq_jakarta')
    if USE_REAL_JAKARTA:
        backend = jakarta
    else:
        backend = QasmSimulator.from_backend(jakarta)



def H_heis3():
    XXs = (I^X^X) + (X^X^I)
    YYs = (I^Y^Y) + (Y^Y^I)
    ZZs = (I^Z^Z) + (Z^Z^I)
    H = XXs + YYs + ZZs
    return H

def U_heis3(t):
    H = H_heis3()
    return (t * H).exp_i()

def evaluate_fidelity(trotter_steps, intermediate_trottor=None, reps=8, shots=8192):

    # by default only evaluate the state after the last trotter step
    # by in order to verify that one indeed do trotterization, we will also evaluate the state in the middle, by specifying `intermediate_trottor` in a range of [1, trotter_steps]
    if intermediate_trottor is None:
        intermediate_trottor = trotter_steps
    target_time = np.pi * intermediate_trottor / trotter_steps

    # The expected final state; necessary to determine state tomography fidelity
    target_state = (U_heis3(target_time) @ (One^One^Zero)).eval().to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)

    Trot_gate, target_qubits = my_trotter(trotter_steps)

    # Initialize quantum circuit for 3 qubits
    qr = QuantumRegister(7)
    qc = QuantumCircuit(qr)

    # Prepare initial state (remember we are only evolving 3 of the 7 qubits on jakarta qubits (q_5, q_3, q_1) corresponding to the state |110>)
    qc.x([3,5])  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)

    # Simulate time evolution under H_heis3 Hamiltonian
    for _ in range(intermediate_trottor):
        qc.append(Trot_gate, [qr[i] for i in target_qubits])

    # Generate state tomography circuits to evaluate fidelity of simulation
    st_qcs = state_tomography_circuits(qc, [qr[1], qr[3], qr[5]])

    jobs = []
    for _ in range(reps):
        job = execute(st_qcs, backend, shots=shots)
        jobs.append(job)

    for job in jobs:
        job_monitor(job, quiet=True)
        try:
            if job.error_message() is not None:
                print(f"[error: trotter_steps={trotter_steps}]", job.error_message())
        except:
            pass

    # Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
    def state_tomo(result, st_qcs):
        # Fit state tomography results
        tomo_fitter = StateTomographyFitter(result, st_qcs)
        rho_fit = tomo_fitter.fit(method='lstsq')
        # Compute fidelity
        fid = state_fidelity(rho_fit, target_state)
        return fid

    # Compute tomography fidelities for each repetition
    fids = []
    for job in jobs:
        fid = state_tomo(job.result(), st_qcs)
        fids.append(fid)

    # print('state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))
    fidelity = np.mean(fids)
    fidelity_stderr = np.std(fids)

    return (fidelity, fidelity_stderr)

if __name__ == "__main__":
    main()
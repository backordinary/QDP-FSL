# https://github.com/Linueks/QuantumComputing/blob/c5876baad39b9337e7e50549f3f1c7c9d3de53dc/IBM-quantum-challenge/src/old_simulation.py
"""
Starting with implementation provided by IBM at:
https://github.com/qiskit-community/open-science-prize-2021/blob/main/ibmq-qsim-challenge.ipynb
@linueks
"""
import numpy as np
import mitiq as mt
import time as time
import qiskit as qk
import qiskit.opflow as opflow
import matplotlib.pyplot as plt
import qiskit.ignis.verification.tomography as tomo
from qiskit.quantum_info import state_fidelity
from decompositions import *
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')



def build_circuit(
    time,
    trotter_step_function,
    trotter_steps=4,
    target_time=np.pi,
    draw_circuit=False,
    n_qubits=7,
    active_qubits=[1, 3, 5],
    symmetry_protection=True,
    transpile_circuit=True,
):
    """
    Function to add specified amount of Trotter steps to a qiskit circuit and
    set correct simulation for each step given desired target time.

    Input:
        time: qiskit.circuit.Parameter
        trotter_step_function: Python Function returning qiskit.circuit
        trotter_steps: int
        target_time: float
        draw_circuit: bool
        show_all_tomography_circuits: bool
        n_qubits: int
        active_qubits: list of qubit positions
        symmetry_protection: bool
        transpile_circuit: int (from 0, 3 according to Qiskit levels)

    Returns:
        quantum_circuit: qiskit.circuit.QuantumCircuit
        quantum_register: qiskit.circuit.QuantumRegister
    #"""
    quantum_register = qk.QuantumRegister(n_qubits)                             # 7 qubits on Jakarta machine. 5 on Belem
    quantum_circuit = qk.QuantumCircuit(quantum_register)

    # yalla fix so I can use the function with other qubit numbers.
    # set up initial state |110>. This assumes you want this initial state.
    quantum_circuit.x([active_qubits[1], active_qubits[2]])                     # Remember to switch back once access to Jakarta
    quantum_circuit.barrier()

    for step in range(1, trotter_steps+1):
        if symmetry_protection and (step)%2 == 1:
            quantum_circuit += symmetry_protection_su_2(
                quantum_register,
                active_qubits,
            )

        single_trotter_step = trotter_step_function(
            time,
            quantum_register,
            active_qubits,
        )
        quantum_circuit += single_trotter_step

        if symmetry_protection and (step)%2 == 1:
            quantum_circuit += symmetry_protection_su_2(
                quantum_register,
                active_qubits,
            )

        #"""
        # I get slightly higher for the best one with this, but it seems so wrong....
        # If I put it after the entire circuit like I thought was described
        # in the paper then the result becomes real bad.
        if symmetry_protection and trotter_steps%2 == 1:
            quantum_circuit += symmetry_protection_step(
                quantum_register,
                active_qubits,
            )
        #"""
        quantum_circuit.barrier()

    if transpile_circuit:
        quantum_circuit = qk.compiler.transpile(
            quantum_circuit,
            basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'reset'],
            optimization_level=transpile_circuit,
        )

    if draw_circuit:
        quantum_circuit.draw(output='mpl')
        plt.tight_layout()
        plt.show()

    #quantum_circuit.measure(measured_qubits, [2, 1, 0])
    quantum_circuit = quantum_circuit.bind_parameters(
        {time: target_time/trotter_steps}
    )

    return quantum_circuit, quantum_register



def generate_tomography_circuits(
    quantum_circuit,
    quantum_register,
    draw_all_tomography_circuits=False,
):
    """
    Generates 27 different quantum circuits to perform complete quantum state
    tomography.

    Inputs:
        quantum_circuit: qiskit.circuit.QuantumCircuit
        quantum_register: qiskit.circuit.QuantumRegister
        draw_all_tomography_circuits: bool

    Returns:
        tomography_circuits: list of qiskit.circuit.QuantumCircuit
    #"""
    measured_qubits = [
        quantum_register[1],
        quantum_register[3],
        quantum_register[5]
    ]                                                                           # Switch back when going to Jakarta
    tomography_circuits = tomo.state_tomography_circuits(
        quantum_circuit,
        measured_qubits
    )
    if draw_all_tomography_circuits:
        for i in range(len(tomography_circuits)):
            tomography_circuits[i].draw(output='mpl')
            plt.tight_layout()
        plt.show()

    return tomography_circuits



def execute_circuit(
    quantum_circuit,
    backend,
    shots,
):
    """
    Putting this into separate function in case I want to add more arguments
    later when optimizing. Looks a bit pointless as of now.
    #"""
    job = qk.execute(quantum_circuit, backend, shots=shots)

    return job



def calculate_fidelity(
    jobs,
    tomography_circuits,
    print_result=True,
):
    ket_zero = opflow.Zero
    ket_one = opflow.One
    final_state_target = ket_one^ket_one^ket_zero
    target_state_matrix = final_state_target.to_matrix()

    fidelities = []

    for job in jobs:
        result = job.result()
        tomography_fitter = tomo.StateTomographyFitter(
            result,
            tomography_circuits,
        )
        rho_fit = tomography_fitter.fit(method='lstsq')
        fidelity = qk.quantum_info.state_fidelity(
            rho_fit,
            target_state_matrix
        )
        fidelities.append(fidelity)

    fidelity_mean = np.mean(fidelities)
    fidelity_std = np.std(fidelities)

    if print_result:
        print(f'state tomography fidelity = {fidelity_mean:.4f}',
               '\u00B1', f'{fidelity_std:.4f}')

    return fidelity_mean, fidelity_std



def run_experiments(
    time,
    backend,
    trotter_step_list,
    min_trotter_steps,
    max_trotter_steps,
    target_time=np.pi,
    shots=8192,
    repetitions=8,
    draw_circuit=True,
    n_qubits=7,
    active_qubits=[1,3,5],
    symmetry_protection=True,
    transpile_circuit=1,
    verbose=True,
):
    """
    Container function to collect and output results for everything interesting
    to calculate. Want to make it run over trotter steps, decompositions,
    repetitions.
    #"""

    n_decompositions = len(trotter_step_list)

    fid_means = np.zeros(
        shape=(
            n_decompositions,
            max_trotter_steps - min_trotter_steps + 1
        )
    )
    fid_stdevs = np.zeros_like(fid_means)

    for i, decomp in enumerate(trotter_step_list):
        if verbose:
            print(repr(decomp), '---------------------------------------------')

        for j, steps in enumerate(range(min_trotter_steps,max_trotter_steps+1)):
            if verbose:
                print(f'Trotter step: {steps}')

            circuit, register = build_circuit(
                time,
                decomp,
                trotter_steps=steps,
                target_time=target_time,
                draw_circuit=draw_circuit,
                n_qubits=n_qubits,
                active_qubits=active_qubits,
                symmetry_protection=True,
                transpile_circuit=True,
            )
            tomography_circuits = generate_tomography_circuits(
                circuit,
                register,
                draw_all_tomography_circuits=False,
            )
            jobs = []
            for k in range(repetitions):
                job = execute_circuit(
                    tomography_circuits,
                    backend,
                    shots,
                )
                jobs.append(job)

                if verbose:
                    print(f'Job ID: {job.job_id()}')
                    qk.tools.monitor.job_monitor(job)

                try:
                    if job.error_message() is not None:
                        print(job.error_message())
                except:
                    pass


            fid_means[i, j], fid_stdevs[i, j] = calculate_fidelity(
                jobs,
                tomography_circuits,
                print_result=True,
            )

    return fid_means, fid_stdevs


if __name__=='__main__':
    provider = qk.IBMQ.load_account()

    provider = qk.IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
    jakarta_backend = provider.get_backend('ibmq_jakarta')
    config = jakarta_backend.configuration()

    #provider = qk.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    #belem_backend = provider.get_backend('ibmq_belem')                          # has the same topology as Jakarta with qubits 1,3,4 corresponding to 1,3,5
    #config = belem_backend.configuration()

    # set up qiskit simulators
    #sim_noisy_belem = qk.providers.aer.QasmSimulator.from_backend(belem_backend)
    #sim_noiseless_belem = qk.providers.aer.QasmSimulator()
    sim_jakarta_noiseless = qk.providers.aer.QasmSimulator()
    sim_noisy_jakarta = qk.providers.aer.QasmSimulator.from_backend(jakarta_backend)

    ket_zero = opflow.Zero
    ket_one = opflow.One
    initial_state = ket_one^ket_one^ket_zero
    time = qk.circuit.Parameter('t')
    shots = 8192
    end_time = np.pi                                                            # Specified in competition
    min_trotter_steps = 4                                                       # 4 minimum for competition
    max_trotter_steps = 12
    trotter_steps = 1                                                           # Variable if just running one simulation
    num_jobs = 2
    symmetry_protection = False

    # should group the two lists here to one dictionary probably
    decompositions = [
        #trotter_step_zyxzyx,
        #trotter_step_zzyyxx,
        #trotter_step_actual_dot_product,
        full_sum_optimal_trotter,
        #trotter_step_dot_product,
        #first_cancellations_zzyyxx,
    ]
    names = [
        #'Trot zyxzyx',
        #'Trot zzyyxx',
        #'Trot x+y z',
        'Trot full sum',
        #'Trot SU1'
        #'Cancel zzyyxx',
    ]

    #"""
    circuit, register = build_circuit(
        time,
        trotter_step_zzyyxx,
        trotter_steps=trotter_steps,
        target_time=end_time,
        draw_circuit=True,
        n_qubits=3,
        active_qubits=[0, 1, 2],
        symmetry_protection=symmetry_protection,
        transpile_circuit=False,
    )
    #"""

    #"""
    active_qubits = [1, 3, 5]
    fid_means, fid_stdevs = run_experiments(
        time,
        sim_noisy_jakarta,
        decompositions,
        min_trotter_steps,
        max_trotter_steps,
        target_time=end_time,
        shots=shots,
        repetitions=num_jobs,
        draw_circuit=False,
        n_qubits=7,
        active_qubits=active_qubits,
        symmetry_protection=symmetry_protection,
        transpile_circuit=3,
        verbose=False,
    )
    np.save(f'../data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}', fid_means)
    np.save(f'../data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}', fid_stdevs)
    #"""

    #fid_means = np.load(f'../data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}.npy')
    #fid_stdevs = np.load(f'../data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_SP{repr(symmetry_protection)}_{len(decompositions)}.npy')

    linestyles = [
        'solid',
        'dotted',
        'dashed',
        'dashdot'
    ]

    x_axis = range(min_trotter_steps, max_trotter_steps+1)

    for i in range(len(decompositions)):
        eb1 = plt.errorbar(
            x_axis, fid_means[i, :],
            yerr=fid_stdevs[i, :],
            errorevery=1,
            label=names[i],
            ls=linestyles[i],
            capsize=5
        )

    plt.xlabel('Trotter Steps')
    plt.ylabel('Fidelity')
    plt.title(f'Trotter Simulation with {shots} Shots, {num_jobs} Jobs, Backend: {config.backend_name}')
    plt.legend()
    plt.savefig(f'../figures/trotter_sim_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_numjobs{num_jobs}_SP{repr(symmetry_protection)}_{len(decompositions)}')
    plt.show()
    #"""

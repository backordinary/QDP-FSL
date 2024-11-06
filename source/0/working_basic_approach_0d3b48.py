# https://github.com/Linueks/QuantumComputing/blob/c5876baad39b9337e7e50549f3f1c7c9d3de53dc/IBM-quantum-challenge/src/working_basic_approach.py
"""
Starting with implementation provided by IBM at:
https://github.com/qiskit-community/open-science-prize-2021/blob/main/ibmq-qsim-challenge.ipynb
"""
import numpy as np
import qiskit as qk
import time as time
import qiskit.opflow as opflow
import matplotlib.pyplot as plt
import qiskit.ignis.verification.tomography as tomo
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import state_fidelity
plt.style.use('seaborn-whitegrid')

#from qiskit.test.mock import FakeJakarta

# this is dumb...
import warnings
warnings.filterwarnings('ignore')



def heisenberg_chain():
    # define Heisenberg XXX hamiltonian opflow way
    # defining operators using qiskit opflow
    identity = opflow.I
    pauli_x = opflow.X
    pauli_y = opflow.Y
    pauli_z = opflow.Z

    x_interaction = (identity^pauli_x^pauli_x) + (pauli_x^pauli_x^identity)
    y_interaction = (identity^pauli_y^pauli_y) + (pauli_y^pauli_y^identity)
    z_interaction = (identity^pauli_z^pauli_z) + (pauli_z^pauli_z^identity)
    total_interaction = x_interaction + y_interaction + z_interaction

    return total_interaction



def propagator(time):
    # define the time evolution operator opflow way
    Hamiltonian = heisenberg_chain()
    time_evolution_unitary = (time * Hamiltonian).exp_i()

    return time_evolution_unitary



def classical_simulation(initial_state):
    # A copy paste from the notebook just to have it here

    time_points = np.linspace(0, np.pi, 100)
    probability_110 = np.zeros_like(time_points)

    for i, t in enumerate(time_points):
        probability_110[i] = np.abs((~initial_state @ propagator(float(t)) \
                                    @ initial_state).eval())**2

    plt.plot(time_points, probability_110)
    plt.xlabel('Time')
    plt.ylabel(r'Probability of state $|110\rangle$')
    plt.title(r'Evolution of state $|110\rangle$ under $H_{XXX}$')
    plt.show()



def construct_trotter_gate_zyxzyx(t, print_subcircuits=False,
                                reverse_circuit=False):
    # decomposition of propagator into quantum gates (copy pasta)
    # I'm not sure I like this way of programming using opflow. Feels like I don't
    # know what's happening behind the scenes. Ask Alessandro

    # build three components of single trotter step: xx, yy, zz (ugly code)
    xx_register = qk.QuantumRegister(2)
    xx_circuit = qk.QuantumCircuit(xx_register, name='xx')
    yy_register = qk.QuantumRegister(2)
    yy_circuit = qk.QuantumCircuit(yy_register, name='yy')
    zz_register = qk.QuantumRegister(2)
    zz_circuit = qk.QuantumCircuit(zz_register, name='zz')

    xx_circuit.ry(np.pi/2, [0,1])
    xx_circuit.cnot(0, 1)
    xx_circuit.rz(2*t, 1)
    xx_circuit.cnot(0, 1)
    xx_circuit.ry(-np.pi/2, [0,1])

    yy_circuit.rx(np.pi/2, [0,1])
    yy_circuit.cnot(0, 1)
    yy_circuit.rz(2*t, 1)
    yy_circuit.cnot(0, 1)
    yy_circuit.rx(-np.pi/2, [0,1])

    zz_circuit.cnot(0, 1)
    zz_circuit.rz(2*t, 1)
    zz_circuit.cnot(0, 1)

    if print_subcircuits:
        print(f'XX-------------------------- \n\n {xx_circuit}')
        print(f'YY-------------------------- \n\n {yy_circuit}')
        print(f'ZZ-------------------------- \n\n {zz_circuit}')

    # Convert custom quantum circuit into a gate
    xx = xx_circuit.to_instruction()
    yy = yy_circuit.to_instruction()
    zz = zz_circuit.to_instruction()

    # Combine subcircuits into a single multiqubit gate representing a single trotter step
    num_qubits = 3

    trot_register = qk.QuantumRegister(num_qubits)
    trot_circuit = qk.QuantumCircuit(trot_register, name='trot zyxzyx')

    for i in range(0, num_qubits-1):
        trot_circuit.append(zz, [trot_register[i], trot_register[i+1]])
        trot_circuit.append(yy, [trot_register[i], trot_register[i+1]])
        trot_circuit.append(xx, [trot_register[i], trot_register[i+1]])

    # Convert custom quantum circuit into a gate
    trotter_gate = trot_circuit.to_instruction()

    if reverse_circuit:
        return trotter_gate.inverse()

    else:
        return trotter_gate



def construct_trotter_gate_zzyyxx(t, reverse_circuit=False):
    xx_register = qk.QuantumRegister(2)
    xx_circuit = qk.QuantumCircuit(xx_register, name='xx')
    yy_register = qk.QuantumRegister(2)
    yy_circuit = qk.QuantumCircuit(yy_register, name='yy')
    zz_register = qk.QuantumRegister(2)
    zz_circuit = qk.QuantumCircuit(zz_register, name='zz')

    xx_circuit.ry(np.pi/2, [0,1])
    xx_circuit.cnot(0, 1)
    xx_circuit.rz(2*t, 1)
    xx_circuit.cnot(0, 1)
    xx_circuit.ry(-np.pi/2, [0,1])

    yy_circuit.rx(np.pi/2, [0,1])
    yy_circuit.cnot(0, 1)
    yy_circuit.rz(2*t, 1)
    yy_circuit.cnot(0, 1)
    yy_circuit.rx(-np.pi/2, [0,1])

    zz_circuit.cnot(0, 1)
    zz_circuit.rz(2*t, 1)
    zz_circuit.cnot(0, 1)

    xx = xx_circuit.to_instruction()
    yy = yy_circuit.to_instruction()
    zz = zz_circuit.to_instruction()


    # Combine subcircuits into a single multiqubit gate representing a single trotter step
    num_qubits = 3

    trot_register = qk.QuantumRegister(num_qubits)
    trot_circuit = qk.QuantumCircuit(trot_register, name='trot zzyyxx')

    trot_circuit.append(zz, [trot_register[0], trot_register[1]])
    trot_circuit.append(zz, [trot_register[1], trot_register[2]])

    trot_circuit.append(yy, [trot_register[0], trot_register[1]])
    trot_circuit.append(yy, [trot_register[1], trot_register[2]])

    trot_circuit.append(xx, [trot_register[0], trot_register[1]])
    trot_circuit.append(xx, [trot_register[1], trot_register[2]])


    # Convert custom quantum circuit into a gate
    trotter_gate = trot_circuit.to_instruction()

    if reverse_circuit:
        return trotter_gate.inverse()

    else:
        return trotter_gate



def generate_circuit(t,
                     trotter_gate_function,
                     trotter_steps=4,
                     target_time=np.pi,
                     draw_circuit=False):
    # generate the full circuit for the trotterized simulation
    # there are also some "fancy / ugly" things happening here
    quantum_register = qk.QuantumRegister(5)                                    # 7 qubits on Jakarta machine. 5 on Belem
    quantum_circuit = qk.QuantumCircuit(quantum_register)

    # set up initial state |110>
    quantum_circuit.x([3, 4])                                                   # Remember to switch back once access to Jakarta
    single_trotter_step = trotter_gate_function(t)

    for n in range(trotter_steps):
        quantum_circuit.append(single_trotter_step,                             # Switch
                            [quantum_register[1],
                             quantum_register[3],
                             quantum_register[4]])

    quantum_circuit = quantum_circuit.bind_parameters(
                        {t: target_time/trotter_steps})

    final_circuit = tomo.state_tomography_circuits(quantum_circuit,         # Switch
                                                [quantum_register[1],
                                                 quantum_register[3],
                                                 quantum_register[4]])

    if draw_circuit:
        print(final_circuit[-1].decompose())
        #print(final_circuit[-1])

    return final_circuit



def execute_circuit(circuit,
                    shots=8192,
                    repetitions=8,
                    backend=QasmSimulator()):
    # wrapper function to run jobs. Assumes QasmSimulator is imported as such.
    jobs = []
    for i in range(repetitions):
        job = qk.execute(circuit, backend, shots=shots)
        print(f'Job ID: {job.job_id()}')
        jobs.append(job)

    for job in jobs:
        qk.tools.monitor.job_monitor(job)

        # this thing seems stupid too
        try:
            if job.error_message() is not None:
                print(job.error_message())
        except:
            pass

    return jobs



def tomography_analysis(result, circuit, target_state):
    # assumes the target state is given as a qiskit opflow state
    target_state_matrix = target_state.to_matrix()
    tomography_fitter = tomo.StateTomographyFitter(result, circuit)
    rho_fit = tomography_fitter.fit(method='lstsq')
    fidelity = qk.quantum_info.state_fidelity(rho_fit, target_state_matrix)

    return fidelity



def run_simulation(time, trotter_gate_function, backend, shots=8192,
                trotter_steps=4, end_time=np.pi, num_jobs=8, draw_circuit=False,
                print_result=True):

    circuit = generate_circuit(time,
                               trotter_gate_function,
                               trotter_steps=trotter_steps,
                               target_time=end_time,
                               draw_circuit=draw_circuit)
    jobs = execute_circuit(circuit, shots=shots, repetitions=num_jobs,
                        backend=backend)
    fidelities = []

    for job in jobs:
        result = job.result()
        fidelity = tomography_analysis(job.result(), circuit, initial_state)
        fidelities.append(fidelity)

    fidelity_mean = np.mean(fidelities)
    fidelity_std = np.std(fidelities)

    if print_result:
        print(f'state tomography fidelity = {fidelity_mean:.4f}',
               '\u00B1', f'{fidelity_std:.4f}')

    return fidelity_mean, fidelity_std


if __name__=='__main__':
    ket_zero = opflow.Zero
    ket_one = opflow.One
    initial_state = ket_one^ket_one^ket_zero

    #classical_simulation(initial_state)
    time = qk.circuit.Parameter('t')

    # set up qiskit simulators
    jakarta_noiseless = QasmSimulator()

    provider = qk.IBMQ.load_account()
    provider = qk.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    #print(provider.backends())
    belem_backend = provider.get_backend('ibmq_belem')                          # has the same topology as Jakarta with qubits 1,3,4 corresponding to 1,3,5
    properties = belem_backend.properties()
    #print(properties)
    config = belem_backend.configuration()
    #print(config.backend_name)
    sim_noisy_belem = QasmSimulator.from_backend(belem_backend)
    #print(sim_noisy_belem)

    #jakarta = provider.get_backend('ibmq_jakarta')
    #properties = jakarta.properties()
    #print(properties)

    # Simulated backend based on ibmq_jakarta's device noise profile
    #sim_noisy_jakarta = QasmSimulator.from_backend(jakarta)

    shots = 1
    trotter_steps = 4                                                           # Variable if just running one simulation
    end_time = np.pi                                                            # Specified in competition
    min_trotter_steps = 1                                                       # 4 minimum for competition
    max_trotter_steps = 1
    num_jobs = 1

    fid_mean = np.zeros(shape=(1, max_trotter_steps - min_trotter_steps + 1))
    fid_std = np.zeros_like(fid_mean)

    #"""
    for i, steps in enumerate(range(min_trotter_steps, max_trotter_steps+1)):
        fid_mean[0, i], fid_std[0, i] = run_simulation(time,
                                                construct_trotter_gate_zyxzyx,
                                                sim_noisy_belem,
                                                shots=shots,
                                                trotter_steps=steps,
                                                end_time=end_time,
                                                num_jobs=num_jobs,
                                                draw_circuit=False)
        """
        fid_mean[1, i], fid_std[1, i] = run_simulation(time,
                                                construct_trotter_gate_zzyyxx,
                                                sim_noisy_belem,
                                                shots=shots,
                                                trotter_steps=steps,
                                                end_time=end_time,
                                                num_jobs=num_jobs,
                                                draw_circuit=False)
        """
    np.save(f'data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}', fid_mean)
    np.save(f'data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}', fid_std)
    #"""

    """
    fid_mean = np.load(f'data/fidelities_mean_{min_trotter_steps}_{max_trotter_steps}_shots{shots}.npy')
    fid_std = np.load(f'data/fidelities_std_{min_trotter_steps}_{max_trotter_steps}_shots{shots}.npy')
    #"""
    eb1 = plt.errorbar(range(min_trotter_steps, max_trotter_steps+1),
                fid_mean[0, :], yerr=fid_std[0, :], errorevery=1,
                label='Trot zyxzyx', ls='--', capsize=5)
    #eb1[-1][0].set_linestyle('--')
    """
    eb2 = plt.errorbar(range(min_trotter_steps, max_trotter_steps+1),
                fid_mean[1, :], yerr=fid_std[1, :], errorevery=1,
                label='Trot zzyykk', ls='-.', capsize=5)
    #eb2[-1][0].set_linestyle('-.')
    """

    plt.xlabel('Trotter Steps')
    plt.ylabel('Fidelity')
    plt.title(f'Trotter Simulation with {shots} Shots, {num_jobs} Jobs, Backend: {config.backend_name}')
    plt.legend()
    plt.savefig(f'figures/trotter_sim_{min_trotter_steps}_{max_trotter_steps}_shots{shots}_numjobs{num_jobs}')
    plt.show()
    """
    circuit = generate_circuit(time,
                               construct_trotter_gate_zzyyxx,
                               trotter_steps=1,
                               draw_circuit=True)
    #"""
    #construct_trotter_gate(time, print_subcircuits=True)

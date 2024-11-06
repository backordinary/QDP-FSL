# https://github.com/Amai-RusseMitsu/discrete-time-crystal-ch/blob/127689538c88aad9d20035f78cf86581e3c546bf/.history/setup/spins_20220430141604.py
import itertools
from qiskit import Aer,IBMQ
from qiskit.providers.aer.noise import NoiseModel

def convert_site(site, N):
    # this function perform the site index conversion as follows:
    # 01234, then convert, e.g. 0 to 4, 1 to 3, etc.
    if N % 2 == 0:
        return int(N/2-site+1)
    elif N % 2 == 1:
        M = (N-1)/2
        return int(M + (M-site))


def spin_combinations(j, N):
    # j is the site index which counts from the l.h.s. in the physical dimension
    # we convert j to ibm_site which counts from the r.h.s.
    combi = list(itertools.product(['0', '1'], repeat=N-1))
    bits = []
    for i in range(2**(N-1)):
        bits.append(''.join(combi[i]))
    spin_up = []
    site = convert_site(j, N)
    for i in range(2**(N-1)):
        spin_up.append(bits[i][:site] + '0' + bits[i][site:])
    spin_down = []
    for i in range(2**(N-1)):
        spin_down.append(bits[i][:site] + '1' + bits[i][site:])
    return spin_up, spin_down


def expectation_value(N, site, qobj, shot_num, circuit):
    spin_up, spin_down = spin_combinations(site, N)
    spin_up_amp = []
    spin_down_amp = []
    for i in range(2**(N-1)):
        spin_up_amp.append(qobj.get_counts(circuit).get(spin_up[i]))
        spin_down_amp.append(qobj.get_counts(circuit).get(spin_down[i]))
    spin_down_amp = [i/shot_num for i in spin_down_amp if i]
    spin_up_amp = [i/shot_num for i in spin_up_amp if i]
    return sum(spin_up_amp) - sum(spin_down_amp)

def initialize_simulator(IsSimulator, api_key, user_hub = None, user_backend = None):
    global provider,backend,coupling_map,basis_gates,simulator
    if IsSimulator == 'Yes':
        simulator = Aer.get_backend('aer_simulator')
    elif IsSimulator == 'No':
        IBMQ.save_account(api_key, overwrite=True)
        provider = IBMQ.load_account()
        provider = IBMQ.get_provider(hub=user_hub)
        backend = provider.get_backend(user_backend)
    elif IsSimulator == 'NoiseLocal':
        IBMQ.save_account(api_key, overwrite=True)
        provider = IBMQ.load_account()
        backend = provider.get_backend('ibmq_lima')
        noise_model = NoiseModel.from_backend(backend)
        # Get coupling map from backend
        coupling_map = backend.configuration().coupling_map
        # Get basis gates from noise model
        basis_gates = noise_model.basis_gates
        simulator = Aer.get_backend('qasm_simulator')
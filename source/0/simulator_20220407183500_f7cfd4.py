# https://github.com/Amai-RusseMitsu/discrete-time-crystal-ch/blob/127689538c88aad9d20035f78cf86581e3c546bf/.history/setup/simulator_20220407183500.py
from mitiq import zne,Executor
from qiskit import execute, QuantumCircuit, Aer, IBMQ
import itertools
from qiskit.providers.aer.noise import NoiseModel
from typing import List
import tqdm


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

def expectation_value(N,site,qobj,shot_num,circuit):
    spin_up,spin_down = spin_combinations(site,N)
    spin_up_amp = []
    spin_down_amp = []
    for i in range(2**(N-1)):
        spin_up_amp.append(qobj.get_counts(circuit).get(spin_up[i]))
        spin_down_amp.append(qobj.get_counts(circuit).get(spin_down[i]))
    spin_down_amp = [i/shot_num for i in spin_down_amp if i]
    spin_up_amp = [i/shot_num for i in spin_up_amp if i]
    return sum(spin_up_amp) - sum(spin_down_amp)


def run_zne(IsSimulator,simulator,backend,shot_num,N,qc_list):

    def batch_run_zne(circuits: List[QuantumCircuit]) -> List[float]:

        def executor(circuit: QuantumCircuit,backend_name: str = "aer_simulator", shots: int = shot_num) -> float:
            if IsSimulator == 'Yes':
                qobj = execute(circuit, simulator,seed_simulator=4, shots = shot_num).result()
            elif IsSimulator == 'No':
                qobj = execute(circuit, backend, shots = shot_num).result()
            
            return expectation_value(N,1,qobj,shot_num,circuit)

        return [zne.execute_with_zne(circuits[j],executor) for j in tqdm(range(len(circuits)))]
    
    return Executor(batch_run_zne, max_batch_size=75).evaluate(qc_list,force_run_all = True)

def run_no_zne(IsSimulator,simulator,backend,shot_num,N,qc_list):

    def batch_run_no_zne(circuits: List[QuantumCircuit]) -> List[float]:

        def executor(self, circuit: QuantumCircuit, backend_name: str = "aer_simulator", shots: int = 5000) -> float:
            if IsSimulator == 'Yes':
                qobj = execute(circuit, simulator,
                                    seed_simulator=4, shots=shot_num).result()
            elif IsSimulator == 'No':
                qobj = execute(circuit, backend,
                                    shots=shot_num).result()
            elif IsSimulator == 'NoiseLocal':
                qobj = execute(circuit, simulator,
                            coupling_map=coupling_map,
                            basis_gates=basis_gates,
                            noise_model=noise_model).result()

            return expectation_value(N, 1, qobj, shot_num, circuit)

        return [executor(circuits[j]) for j in tqdm(range(len(circuits)))]
    
    return Executor(batch_run_no_zne, max_batch_size=75).evaluate(qc_list,force_run_all = True)
    


class Simulator:

    def __init__(self,
                 shot_num=None,
                 IsSimulator=None,
                 api_key=None,
                 user_hub='ibm-q-nus',
                 user_backend=None,
                 N=None,
                 ):
        self.parameters = {
            "shot_num": shot_num,
            "IsSimulator": IsSimulator,
            "api_key": api_key,
            "user_backend": user_backend,
            "N": N,
        }
        self.shot_num = shot_num
        globals()["shot_num"] = shot_num
        self.IsSimulator = IsSimulator
        self.user_backend = user_backend
        self.user_hub = user_hub
        self.N = N
        self.api_key = api_key

    def initialize_simulator(self):
        if self.IsSimulator == 'Yes':
            simulator = Aer.get_backend('aer_simulator')
        elif IsSimulator == 'No':
            IBMQ.save_account(self.api_key, overwrite=True)
            provider = IBMQ.load_account()
            provider = IBMQ.get_provider(hub=self.user_hub)
            backend = self.provider.get_backend(self.user_backend)
        elif self.IsSimulator == 'NoiseLocal':
            IBMQ.save_account(self.api_key, overwrite=True)
            provider = IBMQ.load_account()
            backend = self.provider.get_backend(self.user_backend)
            noise_model = NoiseModel.from_backend(self.backend)
            # Get coupling map from backend
            coupling_map = self.backend.configuration().coupling_map
            # Get basis gates from noise model
            basis_gates = noise_model.basis_gates
            simulator = Aer.get_backend('qasm_simulator')



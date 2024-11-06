# https://github.com/charmerDark/quantumMonteCarlo/blob/9a3575f6b16d4fde1d8c26f739e4eab0e26d1b57/quantum_random.py
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, BasicAer
from qiskit.aqua.components.uncertainty_models import NormalDistribution,UniformDistribution,LogNormalDistribution
from qiskit import IBMQ

def bit_from_counts(counts):
    '''
    function return required counts from the dictionary of counts accquired from the qiskit api
    '''
    return [k for k, v in counts.items() if v == 1][0]
def choice(remote,distribution,apitoken):
    '''
    function sets up quantum circuit of the required number of qubits as required by the ditribution and calls the remote api for IBMQ (or the inbuilt simulator for debugging)
    and acquires results as needed.
    '''
    bits=''
    q = QuantumRegister(10,'q')
    c = ClassicalRegister(10,'c')
    circuit = QuantumCircuit(q,c)
    if distribution=="uniform":
        uniform = UniformDistribution(num_target_qubits = 10,low=- 0, high=1)
        uniform.build(circuit,q)
    elif distribution=='normal':
        normal = NormalDistribution(num_target_qubits = 10, mu=0, sigma=1, low=0, high=1)
        normal.build(circuit,q)
    elif distribution=="lognormal":
        lognorm = LogNormalDistribution(num_target_qubits =10, mu=0, sigma=1, low= 0, high=1)
        lognorm.build(circuit,q)
    circuit.measure(q,c)
    if remote=="True":
                IBMQ.load_account(apitoken)
                provider=IBMQ.get_provider(hub='ibm-q', group='open', project='main') 
                backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= MAX_QUBITS and
                                    not x.configuration().simulator and x.status().operational==True))
                print("least busy backend: ",backend)
    else:
                backend=BasicAer.get_backend('qasm_simulator')
    job_sim = execute(circuit, backend, shots=1)
    sim_result = job_sim.result()
    counts = sim_result.get_counts(circuit)
    bits += bit_from_counts(counts)
    result= int(bits,2)
    if result>=1000:
        result=choice(remote,distribution,apitoken)
        return result
    val=result/1000
    return val

#uantum_random.get_rand_number(lower_bound,upper_bound,remote,distribution,apitoken)
def get_rand_number(min_value, max_value,remote,distribution,apitoken):
    """
    This function gets a random number from a uniform distribution between
    the two input values [min_value, max_value] inclusively
    Args:
    - min_value (float)
    - max_value (float)
    Return:
    - Random number between this range (float)
    """
    range = max_value - min_value
    var=choice(remote,distribution,apitoken)
    return min_value + range*var

if __name__=="__main__":
    '''for i in range(10):
        print(get_rand_number(1,100))
        print("######")'''
        #outdated code above
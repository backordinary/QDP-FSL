# https://github.com/SchrodingersSangru/for_jabroyd/blob/c08e4be5e60f39e3c78ae6494051f498671cba34/credit_risk1.py
import numpy as np


from qiskit import QuantumRegister, QuantumCircuit, Aer, execute
from qiskit.circuit.library import IntegerComparator
from qiskit.utils import QuantumInstance
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem

from qiskit_finance.circuit.library import GaussianConditionalIndependenceModel as GCI
from qiskit.circuit.library import WeightedAdder

from qiskit.circuit.library import LinearAmplitudeFunction

class credit_risk:
    def __init__(self):
        self.n_z = 2
        self.z_max = 2
        self.z_values = np.linspace(-self.z_max, self.z_max, 2**self.n_z)
        self.p_zeros = [0.15, 0.25]
        self.rhos = [0.1, 0.05]
        self.lgd = [1, 2]
        self.K = len(self.p_zeros)
        self.alpha = 0.05
        
    
    def uncerainity_model(self):
        u = GCI(self.n_z, self.z_max, self.p_zeros, self.rhos)
        return u
    
    def get_execute(self):
        job = execute(credit_risk.uncerainity_model, backend=Aer.get_backend("statevector_simulator"))
        return job
    
    def values(self):
        p_z = np.zeros(2**self.n_z)
        p_default = np.zeros(self.K)
        values = []
        probabilities = []
        num_qubits = credit_risk.uncerainity_model.num_qubits
        state = credit_risk.get_execute.result().get_statevector()
        
        if not isinstance(state, np.ndarray):
            state = state.data
        for i, a in enumerate(state):
            # get binary representation
            b = ("{0:0%sb}" % num_qubits).format(i)
            prob = np.abs(a) ** 2

            # extract value of Z and corresponding probability
            i_normal = int(b[-self.n_z:], 2)
            p_z[i_normal] += prob

            # determine overall default probability for k
            loss = 0
            for k in range(self.K):
                if b[self.K - k - 1] == "1":
                    p_default[k] += prob
                    loss += self.lgd[k]
            values += [loss]
            probabilities += [prob]

        values = np.array(values)
        probabilities = np.array(probabilities)

        expected_loss = np.dot(values, probabilities)

        losses = np.sort(np.unique(values))
        pdf = np.zeros(len(losses))
        for i, v in enumerate(losses):
            pdf[i] += sum(probabilities[values == v])
        cdf = np.cumsum(pdf)

        i_var = np.argmax(cdf >= 1 - self.alpha)
        exact_var = losses[i_var]
        exact_cvar = np.dot(pdf[(i_var + 1) :], losses[(i_var + 1) :]) / sum(pdf[(i_var + 1) :])
        return exact_var
    
    
        
    def aggreagtor(self):
        agg = WeightedAdder(self.n_z + self.K, [0] * self.n_z + self.lgd)
        return agg
    
    
    def objective(self):
        
        breakpoints = [0]
        slopes = [1]
        offsets = [0]
        f_min = 0
        f_max = sum(self.lgd)
        c_approx = 0.25

        objective = LinearAmplitudeFunction(
            credit_risk.aggreagtor.num_sum_qubits,
            slope=slopes,
            offset=offsets,
            # max value that can be reached by the qubit register (will not always be reached)
            domain=(0, 2**credit_risk.aggreagtor.num_sum_qubits - 1),
            image=(f_min, f_max),
            rescaling_factor=c_approx,
            breakpoints=breakpoints,
            )
        
        return objective
        
        
        
class prepare_ckt(credit_risk):
    
    def __init__(self):
        
        self.qr_state = QuantumRegister(credit_risk.uncerainity_model.num_qubits, "state")
        self.qr_sum = QuantumRegister(credit_risk.aggreagtor.num_sum_qubits, "sum")
        self.qr_carry = QuantumRegister(credit_risk.aggreagtor.num_carry_qubits, "carry")
        self.qr_obj = QuantumRegister(1, "objective")

    def state_preparation(self):
             
        # define the circuit
        state_preparation = QuantumCircuit(self.qr_state, self.qr_obj, self.qr_sum, self.qr_carry, name="A")

        # load the random variable
        state_preparation.append(credit_risk.uncerainity_model.to_gate(), self.qr_state)

        # aggregate
        state_preparation.append(credit_risk.aggreagtor.to_gate(), self.qr_state[:] + self.qr_sum[:] + self.qr_carry[:])

        # linear objective function
        state_preparation.append(credit_risk.objective.to_gate(), self.qr_sum[:] + self.qr_obj[:])

        # uncompute aggregation
        state_preparation.append(credit_risk.aggreagtor.to_gate().inverse(), self.qr_state[:] + self.qr_sum[:] + self.qr_carry[:])
        
        return state_preparation

    def execute(self):
        job = execute(prepare_ckt.state_preparation, backend=Aer.get_backend("statevector_simulator"))
        return job
    
    def evaluate(self):
        value = 0
        state = prepare_ckt.execute.result().get_statevector()
        print("state is", state)
        if not isinstance(state, np.ndarray):
            state = state.data
        for i, a in enumerate(state):
            b = ("{0:0%sb}" % (len(self.qr_state) + 1)).format(i)[-(len(self.qr_state) + 1) :]
            prob = np.abs(a) ** 2
            if prob > 1e-6 and b[0] == "1":
                value += prob
        return 

    def set_level(self):
        
        epsilon = 0.01
        alpha = 0.05

        qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=100)
        problem = EstimationProblem(
            state_preparation=prepare_ckt.state_preparation,
            objective_qubits=[len(self.qr_state)],
            post_processing=prepare_ckt.execute.post_processing,
        )
        # construct amplitude estimation
        ae = IterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)
        result = ae.estimate(problem)
        
        return result
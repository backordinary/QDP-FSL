# https://github.com/tokarev-i-v/qbyq/blob/468539d606cdeafb39cf08461c5fa62c3b2cade5/src/algorithms/gradient/shift_rule_gradient.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import HGate, XGate, YGate, ZGate, CXGate, RYGate, RZGate
from qiskit import IBMQ, Aer, quantum_info

from qiskit.quantum_info.states import DensityMatrix, Statevector
import numpy as np
p_v = ParameterVector("pv", 3)

class ShiftRuleGradient():
    def __init__(self, cost_function="HST", N = 4, epsilon = 0.5):
        self.cost_function = cost_function
        self.epsilon = epsilon
        self.N = N
        self.tau = 0
        self.nu = 1
        self.unitary_backend = Aer.get_backend('unitary_simulator')

    def get_cost(self, a1, a2):

        two_m = a1.shape[0]
        a1 = np.asmatrix(a1)
        a2 = np.asmatrix(a2)
        a1H = a1.getH()
        m_trace = np.trace(np.matmul(a1H, a2))
        value = ((np.absolute(m_trace)**2)/two_m + 1)/(two_m+1)
        return 1.0 - value

    def get_grad_parameters_values(self, gates_and_parameters):
        """"
            :: gates_and_parameters ::
        """
        gates_array = gates_and_parameters["gates_array"]
        value = np.array([item["parameters_values"] for item in gates_array if item["grad_param"]]).flatten()
        return value
    
    def get_gradient(self, U, gates_and_parameters):
        grad = []
        params_vector = gates_and_parameters["params_vector"]
        qc = gates_and_parameters['qc'].copy()
        parameters_bind_array = gates_and_parameters["params_values"]        
                       
        L = len(params_vector)                
        print("L: ", L)

        for i in range(L):
            qc_minus = qc.copy()
            qc_plus = qc.copy()
            minus_bind_arr = parameters_bind_array.copy()
            minus_bind_arr[i] -= np.pi/2
            plus_bind_arr = parameters_bind_array.copy()
            plus_bind_arr[i] += np.pi/2
            # print("minus_bind_arr:", minus_bind_arr)
            # print("plus_bind_arr:", plus_bind_arr)
            qc_minus = qc_minus.bind_parameters({params_vector: minus_bind_arr})
            qc_plus = qc_plus.bind_parameters({params_vector: plus_bind_arr})
            v_minus = execute(qc_minus, self.unitary_backend).result().get_unitary()
            v_plus = execute(qc_plus, self.unitary_backend).result().get_unitary()
            
            cost_minus = self.get_cost(U, v_minus)
            cost_plus = self.get_cost(U, v_plus)
            
            grad_i_value = 1/2*(cost_plus - cost_minus)
            grad.append(grad_i_value)
            
        return np.array(grad)
    
    def make_qc_from_gates_and_parameters_array(self, gates_and_parameters):
        """
            Creates QuantumCircuit from gates_and_parameters object.
            
            ::gates_and_parameters:: 
        """
        
        params_vector = gates_and_parameters["params_vector"]
        parameters_bind_array = gates_and_parameters["params_values"]
        L = len(params_vector)
        qc = gates_and_parameters["qc"].copy()
        for item in gates_and_parameters["gates_array"]:
            if item["grad_param"]:
                qubits = item["qubits"]
                ind = item["parameters_index"]
                gate = item["gate"](params_vector[ind])
                qc.append(gate, qubits)
            else:
                if item["param"]:
                    gate = item["gate"](*item["parameters_values"])
                else:
                    gate = item["gate"]()
                qubits = item["qubits"]
                qc.append(gate, qubits)
            
        return qc

    def set_parameters_values(self, gates_and_parameters, new_params):
        gates_array = gates_and_parameters["gates_array"]
        for item in gates_array:
            if item["grad_param"]:
                pind = item["parameters_index"]
                item["parameters_values"] = [new_params[pind]]

    def optimize(self, U, gates_and_parameters):
        """
           :: qc :: quantum circuit with optimizable parameters
           :: U :: np.array complex array
           :: gates_and_parameters :: {
               qubits_count: 3,
               params_vector: ParameterVector,
               params_values: np.array,
               gates_array: [
                    {
                        gate: CX,
                        parameters_index: None,
                        parameters_values: None
                        param: False
                        grad_param: False
                    },
                    {
                        gate: ry,
                        parameters_index: [0],
                        parameters_values: [0.5]
                        param: True
                        grad_param: True
                    },
                    {
                        gate: rz,
                        parameters_index: [1],
                        parameters_values: [1.0]
                        param: True
                        grad_param: True
                    },
                    {
                        gate: rz,
                        parameters_values: [1.0],
                        param: True
                        grad_param: False
                    }
                ]
            }
        """
        nu = self.nu
        tau = self.tau
        all_params = []
        count = 0
        gradCount = 0
        params_vector = gates_and_parameters["params_vector"]        
        qc = gates_and_parameters["qc"]
        # print(qc)
    
        alpha_t = gates_and_parameters["params_values"]
        qc_0 = qc.copy()
        qc_0 = qc_0.bind_parameters({params_vector: alpha_t.tolist()})
        V = execute(qc_0, self.unitary_backend).result().get_unitary()
        cost = self.get_cost(U, V)
        all_params.append(alpha_t)
        
        while count < self.N and gradCount < 4:
            tau += 1
            # gradient
            grad = self.get_gradient(U, gates_and_parameters)
            grad_norm = np.linalg.norm(grad)**2
            if grad_norm < self.epsilon:
                gradCount += 1
            # print(f"alpha: {alpha_t}")
            # print(f"grad value: {grad}")
            print(f"grad norm: {grad_norm}")
            alpha_1 = alpha_t - nu * grad
            alpha_2 = alpha_1 - nu * grad
            v_k_a_2 = qc.copy()
            v_k_a_2 = qc.bind_parameters({params_vector: alpha_2.tolist()})
            v_k_a_2 = execute(v_k_a_2, self.unitary_backend).result().get_unitary()
            c_v_k_a_2 = self.get_cost(U, v_k_a_2)

            v_k_a_1 = qc.copy()
            v_k_a_1 = qc.bind_parameters({params_vector: alpha_1.tolist()})
            v_k_a_1 = execute(v_k_a_1, self.unitary_backend).result().get_unitary()
            c_v_k_a_1 = self.get_cost(U, v_k_a_1)
            
            if (cost - c_v_k_a_2) >= nu * grad_norm:
                nu *= 2
                alpha_t = alpha_2
            elif (cost - c_v_k_a_1) < nu*grad_norm/2:
                nu /= 2
                alpha_t = alpha_1
            else:
                alpha_t = alpha_1
            v_k_a_t = qc.copy()
            v_k_a_t = qc.bind_parameters({params_vector: alpha_t.tolist()})
            v_k_a_t = execute(v_k_a_t, self.unitary_backend).result().get_unitary()
            cost = self.get_cost(U, v_k_a_t)
            alpha_opt = alpha_t
            gates_and_parameters["params_values"] = alpha_t
            print(f"Epoch {count}")
            print(f"Opt cost {cost}")
            count += 1
            # if cost > 0.3:
                # return alpha_opt.tolist(), cost 
        return alpha_opt.tolist(), cost
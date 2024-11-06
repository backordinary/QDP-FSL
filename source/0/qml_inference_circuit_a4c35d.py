# https://github.com/andrijapau/qml-thesis-2022/blob/8b348fd17c1daf8b5b7de037577aad056a4eed8b/qml_inference_circuit.py
import qiskit.circuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile
from qiskit.circuit.library import Diagonal, CPhaseGate, RZGate, RYGate
from qiskit.circuit import Qubit
from qiskit.visualization import plot_histogram, plot_gate_map, plot_error_map, plot_coupling_map

from qiskit.providers.ibmq import least_busy
from qiskit import IBMQ

from numpy import array, exp, pi, dot, floor, log2, copy
import matplotlib.pyplot as plt

from utilities import FloatingPointDecimal2Binary


class basis_encoding_circuit:
    """"""

    def __init__(self):
        self.basis_encoding_hf = self.basis_encoding()
        self.algorithms_hf = self.algorithms()
        self.inference_circuit = QuantumCircuit(name='basis_encoding_circuit')
        self.num_of_qubits = 0

        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='strangeworks-hub', group="science-team", project="science-test")

    def encode_data(self, x_vector):
        """"""
        data_num = 0
        for x in x_vector:
            self.basis_encoding_hf.embed_data_to_circuit(x, self.inference_circuit, data_num=data_num)
            data_num += 1

    def add_inner_product_module(self, w_vector, bit_accuracy):
        """"""
        self.inner_prod_reg = QuantumRegister(bit_accuracy, 'anc_ip')
        self.inference_circuit.add_register(self.inner_prod_reg)
        self.algorithms_hf.create_superposition(self.inference_circuit, self.inner_prod_reg)

        num_of_ip_anc = len(self.inner_prod_reg)
        qregs = self.inference_circuit.qregs
        x_qregs = array([reg for reg in qregs if "x" in reg.name], dtype=object)

        if x_qregs.ndim == 2:
            # when data is only bias
            first_element = x_qregs[0, 0]
            last_element = x_qregs[-1, 0]
            start_index = int(first_element.register.name.lstrip('x').rstrip('_float').rstrip('_int'))
            last_index = int(last_element.register.name.lstrip('x').rstrip('_float').rstrip('_int'))

        else:
            first_element = x_qregs[0]
            last_element = x_qregs[-1]
            start_index = int(first_element.name.lstrip('x').rstrip('_float').rstrip('_int'))
            last_index = int(last_element.name.lstrip('x').rstrip('_float').rstrip('_int'))

        x_qregs_sorted = []
        curr = start_index
        for i in range(last_index + 1):
            temp = []
            for x in x_qregs:
                if "{}".format(curr) in x[0].register.name:
                    temp += [x]
            x_qregs_sorted += [temp]
            curr += 1

        w = start_index
        encode_data_circuit = copy(self.inference_circuit.data)
        for x in x_qregs_sorted:
            for type in x:
                for qubit in type:
                    for gate in encode_data_circuit:
                        # gate[1] contains all qubits that have an x gate on them
                        if qubit in gate[1] and gate[0].name == 'x':
                            for i in reversed(range(num_of_ip_anc)):
                                if "int" in qubit.register.name:
                                    self.inference_circuit.append(
                                        CPhaseGate(w_vector[w] * (2 ** qubit.index) * pi / 2 ** i),
                                        [qubit.register[qubit.index], self.inner_prod_reg[i]])
                                if "float" in qubit.register.name:
                                    self.inference_circuit.append(
                                        CPhaseGate(w_vector[w] * (2 ** -(qubit.index + 1)) * pi / 2 ** i),
                                        [qubit.register[qubit.index], self.inner_prod_reg[i]])
            w += 1

        self.algorithms_hf.inv_qft(self.inference_circuit, self.inner_prod_reg)

    def add_activation_fxn_module(self, fxn, bit_accuracy):
        """"""
        self.activation_fxn_reg = QuantumRegister(bit_accuracy, 'anc_fxn')
        self.inference_circuit.add_register(self.activation_fxn_reg)
        self.algorithms_hf.create_superposition(self.inference_circuit, self.activation_fxn_reg)

        def diag_element(z):
            return exp(2 * pi * 1j * fxn(z) / 2 ** bit_accuracy)

        ip_bit_accuracy = len(self.inner_prod_reg)
        diag_array = []
        test_array = []
        for i in range(2 ** ip_bit_accuracy):
            if i < 2 ** (ip_bit_accuracy - 1):
                diag_array.append(diag_element(i))
                test_array.append(i)
            elif i >= 2 ** (ip_bit_accuracy - 1):
                diag_array.append(diag_element(i - 2 ** ip_bit_accuracy))
                test_array.append(i - 2 ** ip_bit_accuracy)

        D_gate = Diagonal(
            array(diag_array)
        )

        for bit in range(bit_accuracy):
            self.inference_circuit.append(D_gate.control(1).power(2 ** bit),
                                          [self.activation_fxn_reg[bit]] + self.inner_prod_reg[:])

        self.algorithms_hf.inv_qft(self.inference_circuit, self.activation_fxn_reg)
        self.measure_register(self.activation_fxn_reg)

    def draw_circuit(self):
        """"""
        self.inference_circuit.draw('mpl', filename='circuit_example.png', style={'name': 'bw', 'dpi': 350})
        plt.show()

    def measure_register(self, register):
        classical_reg = ClassicalRegister(len(register), 'result')
        self.inference_circuit.add_register(classical_reg)
        self.inference_circuit.measure(register, classical_reg)

    def execute_circuit(self, shots, backend=None, optimization_level=None):
        """"""
        self.get_number_of_qubits()

        if backend == None:
            try:
                self.backend = least_busy(
                    self.provider.backends(filters=lambda x: x.configuration().n_qubits >= self.num_of_qubits
                                                             and not x.configuration().simulator
                                                             and x.status().operational == True)
                )
            except:
                print("Not enough qubits")
        else:
            self.backend = self.provider.get_backend(backend)
            job = execute(
                transpile(self.inference_circuit, backend=self.backend, optimization_level=optimization_level),
                backend=self.backend,
                shots=shots
            )
            self.result = job.result()

    def get_counts(self):
        return self.result.get_counts()

    def display_results(self):
        """"""
        plot_histogram(self.result.get_counts(), title="QML Inference Circuit Results", color='black')
        plt.show()

    def get_backend_data(self):
        """"""
        print(self.backend)

    def get_circuit_data(self):
        print(self.inference_circuit.data)

    def get_number_of_qubits(self):
        self.num_of_qubits = self.inference_circuit.num_qubits

    class basis_encoding:
        """"""

        def __init__(self):
            self.int_bin, self.float_bin = '', ''

        def convert_data_to_binary(self, decimal):
            ''''''
            places = 3
            if decimal < 0:
                print("In progress")
            elif decimal > 0:
                x_int, x_float = FloatingPointDecimal2Binary.dec2bin(decimal, places)
                self.int_bin = x_int
                self.float_bin = x_float.rstrip('0')
            else:
                self.int_bin = ''
                self.float_bin = ''

        def embed_data_to_circuit(self, data, circuit, data_num=None):
            """"""
            self.convert_data_to_binary(data)

            if len(self.int_bin) != 0:
                int_reg = QuantumRegister(len(self.int_bin), 'x{}_int'.format(data_num))
                circuit.add_register(int_reg)
                for i in range(len(self.int_bin)):
                    if self.int_bin[i] == '1':
                        circuit.x(int_reg[i])

            if len(self.float_bin) != 0:
                float_reg = QuantumRegister(len(self.float_bin), 'x{}_float'.format(data_num))
                circuit.add_register(float_reg)
                for i in range(len(self.float_bin)):
                    if self.float_bin[i] == '1':
                        circuit.x(float_reg[i])

    class algorithms:
        """"""

        def __init__(self):
            pass

        def create_superposition(self, circuit, register):
            """"""
            circuit.h(register)

        def qft(self, circuit, q_reg):
            self.qft_rotations(circuit, len(q_reg))

        def qft_rotations(self, circuit, qubit_number):
            if qubit_number == 0:
                return circuit
            qubit_number -= 1
            circuit.h(qubit_number)
            for qubit in range(qubit_number):
                circuit.cp(pi / 2 ** (qubit_number - qubit), qubit, qubit_number)

            self.qft_rotations(circuit, qubit_number)

        def inv_qft(self, circuit, q_reg):
            dummy_circuit = QuantumCircuit(len(q_reg), name=r'$QFT^\dagger$')
            self.qft(dummy_circuit, q_reg)
            invqft_circuit = dummy_circuit.inverse()
            circuit.append(invqft_circuit, q_reg)


class amplitude_encoding_circuit:
    def __init__(self):
        self.inference_circuit = QuantumCircuit()
        self.amplitude_encoding_hf = self.amplitude_encoding()
        self.algorithms_hf = self.algorithms()

        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='strangeworks-hub', group="science-team", project="science-test")
        # self.provider = IBMQ.get_provider(hub='ibm-q')

    def encode_data(self, x_vec, w_vec):

        self.num_of_encoding_qubits = -1
        if len(x_vec) == len(w_vec):
            self.num_of_encoding_qubits = floor(log2(len(x_vec)))
        else:
            raise ValueError('ERROR: Size of X and W are not equal.')

        self.U_x = self.amplitude_encoding.U_x__gate(x_vec)
        self.U_w = self.amplitude_encoding.U_w__gate(w_vec)

    def build_circuit(self, qft_bit_accuracy=None):

        U_phi_r = self.U_phi_r__gate()

        self.anc_reg = QuantumRegister(1, name='data_anc')
        self.inference_circuit.add_register(self.anc_reg)

        self.data_reg = QuantumRegister(self.num_of_encoding_qubits, name='data')
        self.inference_circuit.add_register(self.data_reg)

        self.qft_anc_reg = QuantumRegister(qft_bit_accuracy, name='qft_anc')
        self.inference_circuit.add_register(self.qft_anc_reg)

        self.inference_circuit.append(
            U_phi_r,
            self.anc_reg[:] + self.data_reg[:]
        )

        self.algorithms_hf.create_superposition(self.inference_circuit, self.qft_anc_reg)

        self.QPE()

        self.algorithms_hf.inv_qft(self.inference_circuit, self.qft_anc_reg)

    def U_phi_r__gate(self):

        anc_reg = QuantumRegister(1)
        data_reg = QuantumRegister(self.num_of_encoding_qubits)

        U_phi_r__gate__circuit = QuantumCircuit(anc_reg, data_reg)
        U_phi_r__gate__circuit.h(anc_reg)
        U_phi_r__gate__circuit.append(
            self.U_x.control(1, ctrl_state='1'),
            [anc_reg[0], data_reg[:]]
        )
        U_phi_r__gate__circuit.append(
            self.U_w.control(1, ctrl_state='0'),
            [anc_reg[0], data_reg[:]]
        )
        U_phi_r__gate__circuit.h(anc_reg)

        U_phi_r__gate__circuit.draw(output='mpl')
        plt.show()
        return U_phi_r__gate__circuit.to_gate(label=r'$U_{\phi_r}$')

    def G_r__gate(self):

        U_phi_r = self.U_phi_r__gate()
        U_phi_r_inv = U_phi_r.inverse()

        anc_reg = QuantumRegister(1)
        data_reg = QuantumRegister(self.num_of_encoding_qubits)
        G_r__gate__circuit = QuantumCircuit(anc_reg, data_reg)
        G_r__gate__circuit.z(anc_reg)
        G_r__gate__circuit.append(
            U_phi_r_inv,
            anc_reg[:] + data_reg[:]
        )
        G_r__gate__circuit.append(
            RZGate(pi).control(self.num_of_encoding_qubits, ctrl_state='0' * int(self.num_of_encoding_qubits)),
            [anc_reg[:] + data_reg[:-1], data_reg[-1]]
        )
        G_r__gate__circuit.append(
            U_phi_r,
            anc_reg[:] + data_reg[:]
        )
        G_r__gate__circuit.draw(output='mpl')
        plt.show()

        return G_r__gate__circuit.to_gate(label=r'$G_r$')

    def QPE(self):

        G_r = self.G_r__gate()

        for i in range(len(self.qft_anc_reg)):
            self.inference_circuit.append(
                G_r.control(1, ctrl_state='1').power(2 ** i),
                [self.qft_anc_reg[i]] + self.anc_reg[:] + self.data_reg[:]
            )

        # self.inference_circuit.swap(self.qft_anc_reg[0], self.qft_anc_reg[1])

    def add_activation_fxn_module(self, fxn, bit_accuracy):
        """"""
        self.activation_fxn_reg = QuantumRegister(bit_accuracy, 'anc_fxn')
        self.inference_circuit.add_register(self.activation_fxn_reg)
        self.algorithms_hf.create_superposition(self.inference_circuit, self.activation_fxn_reg)

        def diag_element(z):
            return exp(2 * pi * 1j * fxn(z) / 2 ** bit_accuracy)

        D_gate = Diagonal(
            array(
                [diag_element(0), diag_element(1), diag_element(-2), diag_element(-1)]
            )
        )

        for bit in range(bit_accuracy):
            self.inference_circuit.append(D_gate.control(1).power(2 ** bit),
                                          [self.activation_fxn_reg[bit]] + self.qft_anc_reg[:])

        self.algorithms_hf.inv_qft(self.inference_circuit, self.activation_fxn_reg)

    def measure_register(self, register):
        classical_reg = ClassicalRegister(len(register), 'result')
        self.inference_circuit.add_register(classical_reg)
        self.inference_circuit.measure(register, classical_reg)

    def execute_circuit(self, shots, backend=None, optimization_level=None):
        """"""
        # self.measure_register(self.activation_fxn_reg)
        self.measure_register(self.qft_anc_reg)
        # self.inference_circuit.measure_all()
        # if backend == None:
        #     try:
        #         self.backend = least_busy(
        #             self.provider.backends(filters=lambda x: x.configuration().n_qubits >= self.num_of_qubits
        #                                                      and not x.configuration().simulator
        #                                                      and x.status().operational == True)
        #         )
        #     except:
        #         print("Not enough qubits")
        # else:
        self.backend = self.provider.get_backend(backend)
        job = execute(
            transpile(self.inference_circuit, backend=self.backend, optimization_level=optimization_level),
            backend=self.backend,
            shots=shots
        )
        self.result = job.result()
        self.display_results()

    def draw_circuit(self):
        self.inference_circuit.draw(output='mpl')
        plt.show()

    def display_results(self):
        """"""
        plot_histogram(self.result.get_counts(), title="QML Inference Circuit Results", color='black')
        plt.show()

    class amplitude_encoding:
        def __init__(self):
            pass

        def U_x__gate(self):
            x_angle = pi / 2

            return RYGate(2 * x_angle)

        def U_w__gate(self):
            w_angle = 0

            return RYGate(2 * w_angle)

    class algorithms:
        """"""

        def __init__(self):
            pass

        def create_superposition(self, circuit, register):
            """"""
            circuit.h(register)

        def qft(self, circuit, q_reg):
            self.qft_rotations(circuit, len(q_reg))

        def qft_rotations(self, circuit, qubit_number):
            if qubit_number == 0:
                return circuit
            qubit_number -= 1
            circuit.h(qubit_number)
            for qubit in range(qubit_number):
                circuit.cp(pi / 2 ** (qubit_number - qubit), qubit, qubit_number)

            self.qft_rotations(circuit, qubit_number)

        def inv_qft(self, circuit, q_reg):
            dummy_circuit = QuantumCircuit(len(q_reg), name=r'$QFT^\dagger$')
            self.qft(dummy_circuit, q_reg)
            invqft_circuit = dummy_circuit.inverse()
            circuit.append(invqft_circuit, q_reg)

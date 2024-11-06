# https://github.com/ogorodnikov/m1/blob/bfe037084e6f1bfbcc96e1ef6d0203f1ae71945f/a_other/algorithms/shor_independent.py
import math
import fractions
import numpy as np

from qiskit import Aer
from qiskit import execute

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister

from qiskit.circuit import ParameterVector

try:
    from egcd import calculate_egcd
except ModuleNotFoundError:
    from core.algorithms.egcd import calculate_egcd

try:
    from qft import create_qft_circuit
except ModuleNotFoundError:
    from core.algorithms.qft import create_qft_circuit


class Shor:
    
    def run(self, run_values, task_log):
        
        number_input = run_values.get('number')
        base_input = run_values.get('base')
        
        number = int(number_input)
        base = int(base_input)
        
        task_log(f'SHOR number: {number}')
        task_log(f'SHOR base: {base}')
        
        circuit = self.create_shor_circuit(number=number, base=base)
        
        backend = Aer.get_backend('aer_simulator')
        
        job = execute(circuit, backend, shots=1024)
        
        counts = job.result().get_counts()        
        
        run_data = {'Result': {'Counts': counts}, 
                    'Run Values': {'number': number, 'base': base}}
        
        result = self.shor_post_processing(run_data=run_data, task_log=print)


    def create_shor_circuit(self, number, base):

        basic_qubit_count = number.bit_length()
        
        control_qubits_count = basic_qubit_count * 2
        multiplication_qubits_count = basic_qubit_count
        addition_qubits_count = basic_qubit_count + 1
        comparison_qubits_count = 1
        
        measure_bits_count = basic_qubit_count * 2
        
        measurement_bits = list(reversed(range(measure_bits_count)))
        
        control_register = QuantumRegister(control_qubits_count, name="ctrl")
        multiplication_register = QuantumRegister(multiplication_qubits_count, name="mult")
        addition_register = QuantumRegister(addition_qubits_count, name="add")
        comparison_register = QuantumRegister(comparison_qubits_count, name="comp")
        
        measure_register = ClassicalRegister(measure_bits_count, name="meas")
        
        # measure_bits = 

        circuit = QuantumCircuit(control_register, 
                                 multiplication_register, 
                                 addition_register,
                                 comparison_register,
                                 measure_register,
                                 name=f"Shor Circuit")

        circuit.h(control_register)
        circuit.x(multiplication_register[0])

        qft = create_qft_circuit(basic_qubit_count + 1,
                                 flipped=True,
                                 barriers=False)
                                 
        iqft = create_qft_circuit(basic_qubit_count + 1, 
                                  flipped=True, 
                                  inverted=True, 
                                  barriers=False)
        
        final_iqft_circuit = create_qft_circuit(control_qubits_count,
                                                inverted=True,
                                                barriers=False)
        
        phases_count = basic_qubit_count + 1
        phases = self.get_phases(number, phases_count)

        phase_adder = self.create_phase_adder(phases)
        inverted_phase_adder = phase_adder.inverse()
        controlled_phase_adder = phase_adder.control(1)

        for control_qubit_index in range(basic_qubit_count * 2):
            
            base_exponent = 2 ** control_qubit_index
            
            multiplier_uncomputed = self.controlled_modular_multiplication_uncomputed(
                number, 
                base, base_exponent,
                controlled_phase_adder, 
                inverted_phase_adder,
                qft, iqft
            )
            
            control_qubit = control_register[control_qubit_index]
            
            multiplier_uncomputed_qubits = [control_qubit, 
                                           *multiplication_register, 
                                           *addition_register,
                                           *comparison_register]
            
            circuit.append(multiplier_uncomputed, multiplier_uncomputed_qubits)
        
        circuit.append(final_iqft_circuit, control_register)
        
        circuit.measure(control_register, measurement_bits)

        print(f"SHOR circuit:\n{circuit}")
        
        return circuit 
        
       
    def controlled_modular_multiplication_uncomputed(
            self,
            number, 
            base, base_exponent,
            controlled_phase_adder, 
            inverted_phase_adder, 
            qft, iqft
        ):

        # variant: current_base = pow(base, 2**base_exponent, number)
        
        current_base = base ** base_exponent
        
        basic_qubit_count = number.bit_length()
        
        control_register = QuantumRegister(1, "ctrl")
        multiplication_register = QuantumRegister(basic_qubit_count, "mult")
        addition_register = QuantumRegister(basic_qubit_count + 1, "add")
        comparison_register = QuantumRegister(1, "comp")
        
        circuit_name = f"{base}^{base_exponent}*x mod {number}"
        
        circuit = QuantumCircuit(
            control_register, 
            multiplication_register, 
            addition_register, 
            comparison_register, 
            name=circuit_name
        )
        
        controlled_modular_multiplier = self.controlled_modular_multiplication(
            number, 
            current_base,
            controlled_phase_adder, 
            inverted_phase_adder, 
            qft, iqft
        )

        inverted_controlled_modular_multiplier = self.controlled_modular_multiplication(
            number, 
            current_base,
            controlled_phase_adder, 
            inverted_phase_adder,
            qft, iqft,
            inverted=True
        )
        
        multiplier_qubits = [*control_register, 
                             *multiplication_register, 
                             *addition_register, 
                             *comparison_register]
        
        circuit.append(controlled_modular_multiplier, multiplier_qubits)

        for i in range(basic_qubit_count):
            
            circuit.cswap(control_register, 
                          multiplication_register[i],
                          addition_register[i])
            
        circuit.append(inverted_controlled_modular_multiplier, multiplier_qubits)
        
        # print(f"SHOR controlled_modular_multiplication_uncomputed:\n{circuit}")
        
        return circuit
        
        
    def controlled_modular_multiplication(self,
                                          number, 
                                          current_base,
                                          controlled_phase_adder, 
                                          inverted_phase_adder, 
                                          qft, iqft,
                                          inverted=False):

        basic_qubit_count = number.bit_length()

        phase_parameters = ParameterVector("phases", length=basic_qubit_count + 1)
        
        double_controlled_modular_adder = self.double_controlled_modular_adder(
            phase_parameters, 
            controlled_phase_adder, 
            inverted_phase_adder, 
            qft, iqft
        )
        
        if inverted:
            base_inverse = self.modular_multiplicative_inverse(
                base=current_base, 
                modulus=number)
            actual_base = base_inverse
            actual_adder = double_controlled_modular_adder.inverse()
            
        else:
            actual_base = current_base
            actual_adder = double_controlled_modular_adder
        
        control_register = QuantumRegister(1, "ctrl")
        multiplication_register = QuantumRegister(basic_qubit_count, "mult")
        addition_register = QuantumRegister(basic_qubit_count + 1, "add")
        comparison_register = QuantumRegister(1, "comp")
        
        circuit_name = "Inverted " * inverted + "Controlled Modular Multiplication"
        
        circuit = QuantumCircuit(
            control_register, 
            multiplication_register, 
            addition_register, 
            comparison_register, 
            name=circuit_name
        )
        
        circuit.append(qft, addition_register)

        for multiplication_qubit_index in range(basic_qubit_count):
            
            # variant: partial_coefficient = 2 ** multiplication_qubit_index
            
            partial_coefficient = pow(2, multiplication_qubit_index, number)
            
            partial_base = (actual_base * partial_coefficient) % number
            
            phases = self.get_phases(partial_base, basic_qubit_count + 1)
            
            adder = actual_adder.assign_parameters({phase_parameters: phases})
            
            adder_qubits = [*control_register, 
                            multiplication_register[multiplication_qubit_index], 
                            *addition_register, 
                            *comparison_register]
            
            circuit.append(adder, adder_qubits)
            
        circuit.append(iqft, addition_register)
        
        # print(f"SHOR controlled_modular_multiplication:\n{circuit}")
        
        return circuit


    def double_controlled_modular_adder(self, 
                                        phase_parameters, 
                                        controlled_phase_adder, 
                                        inverted_phase_adder, 
                                        qft, iqft):
        
        control_register = QuantumRegister(1, "ctrl")
        mult_register = QuantumRegister(1, "mult")
        add_register = QuantumRegister(len(phase_parameters), "add")
        comparison_register = QuantumRegister(1, "comp")
        
        circuit = QuantumCircuit(
            control_register, 
            mult_register, 
            add_register, 
            comparison_register,
            name="Double Controlled Modular Adder"
        )

        adder_with_parameters = self.create_phase_adder(phase_parameters)
        double_controlled_adder = adder_with_parameters.control(2)
        
        inverted_double_controlled_adder = double_controlled_adder.inverse()
        
        
        # compute modular addition

        circuit.append(double_controlled_adder, [*control_register, 
                                                 *mult_register, 
                                                 *add_register])

        circuit.append(inverted_phase_adder, add_register)
        circuit.append(iqft, add_register)
        
        circuit.cx(add_register[-1], comparison_register[0])
        
        circuit.append(qft, add_register)
        circuit.append(controlled_phase_adder, [*comparison_register, 
                                                *add_register])
        
        # uncompute modular addition

        circuit.append(inverted_double_controlled_adder, [*control_register, 
                                                          *mult_register, 
                                                          *add_register])
        circuit.append(iqft, add_register)
        
        circuit.x(add_register[-1])
        circuit.cx(add_register[-1], comparison_register[0])
        circuit.x(add_register[-1])
        
        circuit.append(qft, add_register)
        circuit.append(double_controlled_adder, [*control_register, 
                                                 *mult_register, 
                                                 *add_register])
        
        # print(f"SHOR double_controlled_modular_adder:\n{circuit}")
        
        return circuit
        

    def create_phase_adder(self, phases):
        
        qubits_count = len(phases)
        
        phase_adder_circuit = QuantumCircuit(qubits_count, name="Phase adder")
        
        for i, phase in enumerate(phases):
            phase_adder_circuit.p(phase, i)
            
        # print(f"SHOR phase_adder_circuit:\n{phase_adder_circuit}")
        
        return phase_adder_circuit
        

    def get_phases(self, number, phases_count):
        
        number_bits = bin(int(number))[2:]
        number_bits_filled = number_bits.zfill(phases_count)
        number_bits_reversed = reversed(number_bits_filled)
        digits = list(map(int, number_bits_reversed))
        
        angles = np.zeros(phases_count)
        
        for i in range(len(digits)):
            angle = 0
            
            for j, digit in enumerate(digits[:i + 1]):
                delta = j - i
                angle += digit * 2 ** delta
                
            angles[i] = angle
            
        phases = angles * np.pi

        # print(f"SHOR number: {number}")
        # print(f"SHOR digits: {digits}") 
        # print(f"SHOR phases {phases}")
        
        return phases
        

    def modular_multiplicative_inverse(self, base, modulus):
        
        greatest_common_divisor, bezout_s, bezout_t = calculate_egcd(base, modulus)
        
        # print(f"SHOR base: {base}")        
        # print(f"SHOR modulus: {modulus}")        
        # print(f"SHOR greatest_common_divisor: {greatest_common_divisor}")        
        # print(f"SHOR bezout_s: {bezout_s}")        
        # print(f"SHOR bezout_t: {bezout_t}")        
        
        if greatest_common_divisor != 1:
            raise ValueError(f"Modular inverse does not exist")
                
        return bezout_s % modulus

    
    def shor_post_processing(self, run_data, task_log):
    
        number_str = run_data['Run Values']['number']
        base_str = run_data['Run Values']['base']
        counts = run_data['Result']['Counts']
        
        number = int(number_str)
        base = int(base_str)
        
        states = list(counts)
        qubits_count = len(states[0])
    
        task_log(f'SHOR run_data: {run_data}')
    
        task_log(f'SHOR number: {number}')
        task_log(f'SHOR base: {base}')
        task_log(f'SHOR counts: {counts}')
        
        task_log(f'SHOR states: {states}')
        task_log(f'SHOR qubits_count: {qubits_count}')
        
        orders = []
        
        for state in states:
            
            state_binary = int(state, 2)
            
            phase = state_binary / 2 ** qubits_count
            
            phase_fraction = fractions.Fraction(phase).limit_denominator(15)
            
            order = phase_fraction.denominator
            
            orders.append(order)
            
            task_log()
            task_log(f'SHOR state: {state}')
            task_log(f'SHOR state_binary: {state_binary}')
            task_log(f'SHOR phase: {phase}')
            task_log(f'SHOR phase_fraction: {phase_fraction}')
            task_log(f'SHOR order: {order}')
        
        factors = set()
        
        for order in orders:
            
            factor_p1 = math.gcd(base ** (order // 2) - 1, number)
            factor_p2 = math.gcd(base ** (order // 2) + 1, number)
            
            factor_q1 = number // factor_p1
            factor_q2 = number // factor_p2
            
            # task_log(f'SHOR factor_p1: {factor_p1}')
            # task_log(f'SHOR factor_p2: {factor_p2}')
            # task_log(f'SHOR factor_q1: {factor_q1}')
            # task_log(f'SHOR factor_q2: {factor_q2}')
            
            factors.add(factor_p1)
            factors.add(factor_p2)
            factors.add(factor_q1)
            factors.add(factor_q2)

        non_trivial_factors = factors - {1, number}
        
        task_log()
        task_log(f'SHOR orders: {orders}')   
        task_log(f'SHOR factors: {factors}')    
        task_log(f'SHOR non_trivial_factors: {non_trivial_factors}')
        
        return {'Factors': non_trivial_factors}



if __name__ == '__main__':
    
    # run_values = {'number': '15', 'base': '2'}
    run_values = {'number': '2', 'base': '2'}
    
    Shor().run(run_values, print)

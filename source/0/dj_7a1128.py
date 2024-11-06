# https://github.com/attianopp/CS239---Quantum-Programming/blob/ce61f0646f01b7785ae5dfa06144417e2d037f89/qiskit/nishant/dj.py
#!/usr/bin/python

import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator
from itertools import combinations
import time, random
import matplotlib.pyplot as plt
from collections.abc import Iterable
import math

BALANCED=0
CONSTANT=1
NAMES={0: "balanced", 1: "constant"}

class Program:
    def run_dj(self, f, num_shots=1000):
        """
        High level function to run Deutsch Jozsa. This use f to generate Uf, then builds a circuit, runs it, and measures and returns if the function is balanced.
        f is assumed to be balanced or constant. If it measures 0^n, f is constant
        Args: 
        f: input function represented by 2^n array where each entry f[i] represents f applied to that row indexf(i)
        num_shots: number of times to measure
        Returns:
        time_elapsed, answer: time taken to compile and measure. 0 if f is balanced or 1 o.w.
        """
        n = int(math.log(len(f), 2))
        start = time.time()
        # create a gate from f
        uf = self.create_uf(n, f)
        # build the dj circuit
        circuit = self.build_circuit(n, uf)
        # measure the circuit
        results = self.measure(circuit, num_shots=num_shots)
        end = time.time()
        time_elapsed = int((end - start) * 1000)
        # return constant if 0^n was measured
        return time_elapsed, self.evaluate(results, n)
        
    def build_circuit(self, n, uf):
        """
        Assemble the circuit. Create a circuit of n+1 qubits mapped to n bits. Flip helper bit to 1. Apply H to all. Apply Uf to all. Then H to all the n qubits and measure.
        Args: 
        n: number of bits - arity of f
        uf: Uf gate representation of f
        num_shots: number of times to measure
        Returns:
        qiskit circuit implement DJ
        """
        if not isinstance(uf, Operator):
            raise ValueError("uf must be an operator!")
        if not isinstance(n, int):
            raise ValueError("n must be an int")
        
        num_qubits = n + 1

        # |00...0>
        circuit = QuantumCircuit(num_qubits, n)

        # |00...1>
        circuit.x(num_qubits - 1)

        # H^nH |00...1>
        for qubit in range(num_qubits):
            circuit.h(qubit)

        # Uf H^nH |00...1>
        # Reverse the inputs for the endian
        circuit.append(uf, range(num_qubits))
        
        # H^nI Uf H^nH |00...1>
        for qubit in range(num_qubits - 1):
            circuit.h(qubit)
        
        circuit.measure(range(n), range(n))
        return circuit
    def measure(self, circuit, backend='qasm_simulator', num_shots=1000):
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("circuit must be a QuantumCircuit")
        if not isinstance(backend, str):
            raise ValueError("backend must be of type string")
        if not isinstance(num_shots, int):
            raise ValueError("num_shots must be of type int")
        """
        Measure the circuit and return the output
        Args: 
        circuit: the qiskit circuit
        backend: what to run the ciruit it on
        num_shots: number of times to measure
        Returns:
        counts: map of measurement to count
        """
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("circuit must be a QuantumCircuit")
        if not isinstance(backend, str):
            raise ValueError("backend must be of type string")
        if not isinstance(num_shots, int):
            raise ValueError("num_shots must be of type int")
        
        simulator = Aer.get_backend(backend)
        job = execute(circuit, simulator, shots=num_shots)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts
    def evaluate(self, counts, n):
        """
        Evaluate the counts. Return constant if 0^n is measured at all.
        Args: 
        counts: map of measurement to count
        n: number of bits - arity of f        
        Returns:
        result: CONSTANT (0) if 0^n was measured at all, o.w. BALANCED (1)
        """
        if not isinstance(counts, dict):
            raise ValueError("counts must be of type dict")
        if not isinstance(n, int):
            raise ValueError("n must be of type dict")
        zero = '0'*n
        if zero in counts and len(counts) != 1:
            # 0^n and other results should never happen
            raise ValueError("Uh oh! multiple results: %s" % (counts))
        elif zero in counts:
            # if f is constant 0^n is measured with 100% probability
            return CONSTANT
        else:
            return BALANCED
    def create_uf(self, n, f):
        """
        Wrapper function to create Uf. Build the matrix and make it a gate
        Args: 
        n: number of bits - arity of f
        f: input function represented by 2^n array where each entry f[i] represents f applied to that row indexf(i)
        Returns:
        gate: gate representing uf
        """
        if not isinstance(n, int):
            raise ValueError("n must be an int")
        
        matrix = self.build_matrix(n, f)
        uf = Operator(matrix)
        return uf
    def build_matrix(self, n, f):
        """
        Convert f to a 2^(n+1) x 2^(n+1) matrix. Care must be taken since qiskit is little endian.
        for all x in {0, 1)^n, b in 0,1, calculate x (f(x) + b), place a 1 at matrix[2^(n+1) - xb][2^(n+1) - x (f(x) + b)]
        Args: 
        n: number of bits - arity of f
        f: input function represented by 2^n array where each entry f[i] represents f applied to that row indexf(i)
        Returns:
        matrix: matrix representing the uf gate
        """
        if not isinstance(f, Iterable):
            raise ValueError("f must be a 1d array of 1s and 0s")
        num_qubits = n + 1
        matrix = [[ 0 for _ in range(2**num_qubits) ] for _ in range(2**num_qubits) ]
        for i in range(0, 2**num_qubits):
            xb = self.bit_string(i, num_qubits)
            x = xb[:-1]
            b = xb[-1]
            fx = f[self.to_int(x)]
            if fx != 0 and fx != 1:
                raise ValueError("f must be a 1d array of 0s and 1s")
            xbfx = x + [b ^ fx]
            # measure from end to account for little endian
            matrix[self.to_int(xb[::-1])][self.to_int(xbfx[::-1])] = 1
        return matrix
    def to_int(self, bit_string):
        """
        Convert a bit string (list of ints) to an int
        Args: 
        bit_string: list of 1s or 0s
        Returns:
        num: integer of binary represantation
        """
        
        return int("".join([str(num) for num in bit_string]), 2)
    def bit_string(self, i, n):
        """
        Convert an int to a bit string of length n
        Args: 
        i: number to convert
        n: desired length
        Returns:
        binary representation of i
        """
        
        return [ int(char) for char in bin(i)[2:].zfill(n) ]
    def generate_functions(self, n, fully_random=False, random_trials=100):
        """
        Generate test functions of input length n. If fully random is false, generate all possible constant and balanced function.
        For n > 3 or fully_random = True, generate m random trials where m = random_trials
        Args: 
        n: bit length
        fully_random: select random trials for n<=3
        random_trials: number of random functions to generate when selecting at random.
        Returns:
        list of tuples, first item is BALANCED/CONSTANT and second is the function
        """
        
        N = 2**n
        functions = []
        functions.append((CONSTANT, [0]*N))
        functions.append((CONSTANT, [1]*N))
        if not fully_random and n <= 3:
            for combination in combinations(range(N), N // 2):
                ans = [0]*N
                for c in combination:
                    ans[c] = 1
                functions.append((BALANCED, ans))
        else:
            for _ in range(random_trials):
                ans = [0]*N
                indices = random.sample(range(N), N//2)
                for index in indices:
                    ans[index] = 1
                functions.append((BALANCED, ans))
        return functions
    def run_tests(self, max_bits):
        """
        Test for 1 to max_bits. For 1 to 3, test all possible input functions. After that, select 100 at random.
        Args: 
        max_bits: number of bits to test up to
        """
        
        for n in range(1, max_bits + 1):
            print("Running exhaustive tests for %d bits" % n)
            functions = self.generate_functions(n)
            for expected, f in functions:
                run_time, calculated = self.run_dj(f)
                if calculated != expected:
                    raise ValueError("Test failed: (%s). Got: %d. Expected %d." % (f, calculated, expected))
                print(len(f))
                print("Test passed for %s function. Got: %d. Expected %d. Tooks %d ms." % (NAMES[calculated], calculated, expected, run_time))
            print("%d tests passed." % (len(functions)))
    def collect_data(self, max_bits, num_trials, npz_filename):
        """
        Generate benchmarking data.
        from 1 to max_bits, save the time for num_trials.
        Run 100 trials for n = 6 random ufs and collect time
        Args: 
        max_bits: number of bits to test up to
        num_trials: how many trials for each n
        npz_filename: where tp save output
        """
        
        time_arr = np.zeros([max_bits, num_trials])

        for n in range(1, max_bits+1):
            print("Starting benchmarking for n = %d" % (n))
            functions = self.generate_functions(n, True, num_trials)
            for trial in range(num_trials):
                expected, f = functions[trial]
                run_time, calculated = self.run_dj(f, 1)
                if calculated != expected:
                    raise ValueError("Test failed: (%s). Got: %d. Expected %d." % (f, calculated, expected))
                print("Test passed for %s function. Got: %d. Expected %d. Tooks %d ms." % (NAMES[calculated], calculated, expected, run_time))
                time_arr[n-1, trial] = run_time / 1000.0


        num_times = 100
        variance_arr = np.zeros(num_times)
        n = 9
        print("Starting benchmarking for different Ufs for n = %d" % (n))
        functions = self.generate_functions(n, True, num_times)
        for trial in range(num_times):
            expected, f = functions[trial]
            run_time, calculated = self.run_dj(f, 1)
            if calculated != expected:
                raise ValueError("Test failed: (%s). Got: %d. Expected %d.", f, calculated, expected)
            print("Test passed for %s function. Got: %d. Expected %d. Tooks %d ms." % (NAMES[calculated], calculated, expected, run_time))
            variance_arr[trial] = run_time / 1000.0
        np.savez(npz_filename, time_arr = time_arr, n_list = list(range(1, max_bits+1)), n = n, num_times = num_times, variance_arr = variance_arr)

    def plot_data(self, npz_filename):
        """
        Plot benchmarking data. Plot average run time for each n and variance data for 100 random Ufs.
        Args: 
        npz_filename: where tp load data
        """
        
        data = np.load(npz_filename)

        time_arr = data['time_arr']
        n_list = data['n_list']

        avg_time = np.sum(time_arr,1)/np.shape(time_arr)[1]

        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        z = np.polyfit(n_list, avg_time, 10)
        p = np.poly1d(z)

        plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
        plt.plot(n_list, avg_time, ls = '', markersize = 15, marker = '.',label = 'M')
        plt.title('Execution time scaling for Deutsch-Jozsa algorithm', fontsize=25)
        plt.xlabel('n (bit string length)',fontsize=20)
        plt.ylabel('Average time of execution (s)',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        fig.savefig('figures/dj.png', bbox_inches='tight')

        num_times = data['num_times']
        variance_arr = data['variance_arr']
        n = data['n']

        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))


        plt.hist(variance_arr)
        plt.title('Dependence of execution time on $U_f$ (Deutsch-Jozsa algorithm)', fontsize=25)
        plt.xlabel('Execution time (s)',fontsize=20)
        plt.ylabel('Frequency of occurence',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)


        fig.savefig('figures/dj_hist.png', bbox_inches='tight')
        
        
if __name__ == "__main__":
    p = Program()
    p.run_tests(8)
    #p.collect_data(10, 5, "dj.npz")
    #p.plot_data("dj.npz")

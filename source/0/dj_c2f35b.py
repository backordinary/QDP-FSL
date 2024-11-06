# https://github.com/attianopp/CS239---Quantum-Programming/blob/ce61f0646f01b7785ae5dfa06144417e2d037f89/ibmq/submission/dj.py
#!/usr/bin/python

import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
    Aer,
    IBMQ)
from qiskit.visualization import plot_histogram
from qiskit.compiler import transpile, assemble
from qiskit.quantum_info.operators import Operator
from itertools import combinations
import time, random
import matplotlib.pyplot as plt
from collections.abc import Iterable
import math

BALANCED=0
CONSTANT=1
NAMES={0: "balanced", 1: "constant"}

API_TOKEN = "740ea7965ab0da8ae592c503e260b98f80bc59414c8668a7b2c86bed0198dc5bca8846ae46c098881bba48f8019b0bdfc3afe79a2b57289d3f1ec21cac8b511d"


class Program:
    def run(self, f, backend=None, num_shots=1000):
        """
        High level function to run Deutsch Jozsa. This use f to generate Uf, then builds a circuit, runs it, and measures and returns if the function is balanced.
        f is assumed to be balanced or constant. If it measures 0^n, f is constant
        Args: 
        f: input function represented by 2^n array where each entry f[i] represents f applied to that row indexf(i)
        num_shots: number of times to measure
        backend: backend object to run on
        Returns:
        counts dictionary, most frequent measurement, compilation time run time
        """
        if backend == None:
            provider = IBMQ.enable_account(API_TOKEN)
            backend = provider.backends.ibmq_burlington
        
        n = int(math.log(len(f), 2))
        start = time.time()
        # create a gate from f
        uf = self.create_uf(n, f)
        # build the dj circuit
        circuit = self.build_circuit(n, uf)
        # measure the circuit
        counts, assembly_time, run_time = self.measure(circuit, backend, num_shots=num_shots)
        end = time.time()
        time_elapsed = int((end - start) * 1000)
        return counts, self.evaluate(counts), assembly_time, run_time
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
        
    def build_circuit(self, n, uf, flag=False, custom_uf=None):
        """
        Assemble the circuit. Create a circuit of n+1 qubits mapped to n bits. Flip helper bit to 1. Apply H to all. Apply Uf to all. Then H to all the n qubits and measure.
        Args: 
        n: number of bits - arity of f
        uf: Uf gate representation of f
        num_shots: number of times to measure
        Returns:
        qiskit circuit implement DJ
        """
        if not isinstance(n, int):
            raise ValueError("n must be an int")
        
        num_qubits = n + 1

        # |00...0>
        circuit = QuantumCircuit(num_qubits, n)

        # |00...1>
        circuit.x(num_qubits - 1)
        circuit.barrier()
        
        # H^nH |00...1>
        for qubit in range(num_qubits):
            circuit.h(qubit)
        circuit.barrier()
        # Uf H^nH |00...1>
        # Reverse the inputs for the endian
        if flag == False:
            if not np.array_equal(uf._data, np.identity(2**num_qubits, dtype=np.complex64)):
                circuit.append(uf, range(num_qubits))
            else:
                print("Skipping uf since it's the identity")
        else:
            circuit += custom_uf
        circuit.barrier()
        # H^nI Uf H^nH |00...1>
        for qubit in range(num_qubits - 1):
            circuit.h(qubit)
        circuit.barrier()
        circuit.measure(range(n), range(n))
        return circuit
    def get_uf(self, n, circuit_type):
        """
        generate a random uf operator
        n: number of input bits
        circuit_type: str: constant0 for f(x) = 0, constant1 for f(x) = 1, balanced for random balanced 
        returns: operator
        """
        num_qubits = n + 1
        circuit = QuantumCircuit(num_qubits, n)
        if circuit_type == "constant0":
            pass
        elif circuit_type == "constant1":
            circuit.x(n - 1)
        else:
            r = random.randint(0, 2**n - 1)
            for index in range(0, n):
                if (1 << index) & r != 0:
                    circuit.x(index)
            circuit.barrier()
            for index in range(0, n):
                circuit.cx(index, num_qubits - 1)
            circuit.barrier()
            for index in range(0, n):
                if (1 << index) & r != 0:
                    circuit.x(index)
        return circuit
            
    def measure(self, circuit, backend, num_shots=1000):
        """
        Measure the circuit and return the output
        Args: 
        circuit: the qiskit circuit
        backend: what to run the ciruit it on
        num_shots: number of times to measure
        Returns:
        counts: map of measurement to count
        """
        start = time.time()
        qobj = assemble(transpile(circuit, backend=backend, optimization_level=3), backend=backend, shots=num_shots)
        end = time.time()
        assembly_time = int((end - start) * 1000)
        job = backend.run(qobj)
        job = backend.retrieve_job(job.job_id())
        while job._api_status != 'COMPLETED':
            if "ERROR" in job._api_status:
                raise ValueError("Job status is %s: " % job._api_status)
            print('Current job status: %s' % job._api_status)
            time.sleep(5)
            job = backend.retrieve_job(job.job_id())
        return job.result().get_counts(circuit), assembly_time/1000.0, job.result().time_taken
    def evaluate(self, counts):
        """
        Evaluate the counts. Return constant if 0^n is measured at all.
        Args: 
        counts: map of measurement to count
        Returns:
        result: returns most measured element
        """
        if not isinstance(counts, dict):
            raise ValueError("counts must be of type dict")
        max_key = None
        for key in counts:
            if max_key == None or (counts[key] > counts[max_key]):
                max_key = key
        return max_key
    def build_custom_circuit(self, n, circuit_type):
        """
        Assemble the circuit. Create a circuit of n+1 qubits mapped to n bits. Flip helper bit to 1. Apply H to all. Apply Uf to all. Then H to all the n qubits and measure.
        Args: 
        n: number of bits - arity of f
        uf: Uf gate representation of f
        num_shots: number of times to measure
        Returns:
        qiskit circuit implement DJ
        """
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

        if circuit_type == "balanced0":
            pass
        elif circuit_type == "balanced1":
            pass
        if not np.array_equal(uf._data, np.identity(2**num_qubits, dtype=np.complex64)):
            circuit.append(uf, range(num_qubits))
        else:
            print("Skipping uf since it's the identity")
        
        # H^nI Uf H^nH |00...1>
        for qubit in range(num_qubits - 1):
            circuit.h(qubit)
        
        circuit.measure(range(n), range(n))
        return circuit

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
        #functions.append((CONSTANT, [0]*N))
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
    def build_histogram(self):
        provider = IBMQ.enable_account(API_TOKEN)
        backend = provider.backends.ibmq_burlington
        f = [1] * (2**1)
        counts, _, _, _ = self.run(f, backend, num_shots=8192)
        
        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        answers = [key[::-1] for key in counts.keys()]
        answers.sort()
        frequencies = [ counts[answer[::-1]] for answer in answers ]
        
        plt.bar(range(len(counts)), frequencies, align='center', tick_label=answers)
        plt.title('Results of f(x) = 1 for n = %d (Deutsch-Jozsa algorithm)' % 1, fontsize=25)
        plt.xlabel('Measurement',fontsize=20)
        plt.ylabel('Frequency of occurence',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        
        fig.savefig('figures/dj_counts_histogram_%d.png' % 1, bbox_inches='tight')
        f = [1] * (2**2)
        counts, _, _, _ = self.run(f, backend, num_shots=8192)
        
        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        answers = [key[::-1] for key in counts.keys()]
        answers.sort()
        frequencies = [ counts[answer[::-1]] for answer in answers ]
        
        plt.bar(range(len(counts)), frequencies, align='center', tick_label=answers)
        plt.title('Results of f(x) = 1 for n = %d (Deutsch-Jozsa algorithm)' % 2, fontsize=25)
        plt.xlabel('Measurement',fontsize=20)
        plt.ylabel('Frequency of occurence',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        
        fig.savefig('figures/dj_counts_histogram_%d.png' % 2, bbox_inches='tight')
        f = [1] * (2**3)
        counts, _, _, _ = self.run(f, backend, num_shots=8192)
        
        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        answers = [key[::-1] for key in counts.keys()]
        answers.sort()
        frequencies = [ counts[answer[::-1]] for answer in answers ]
        
        plt.bar(range(len(counts)), frequencies, align='center', tick_label=answers)
        plt.title('Results of f(x) = 1 for n = %d (Deutsch-Jozsa algorithm)' % 3, fontsize=25)
        plt.xlabel('Measurement',fontsize=20)
        plt.ylabel('Frequency of occurence',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        
        fig.savefig('figures/dj_counts_histogram_%d.png' % 3, bbox_inches='tight')
        f = [1] * (2**4)
        counts, _, _, _ = self.run(f, backend, num_shots=8192)
        
        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        answers = [key[::-1] for key in counts.keys()]
        answers.sort()
        frequencies = [ counts[answer[::-1]] for answer in answers ]
        
        plt.bar(range(len(counts)), frequencies, align='center', tick_label=answers)
        plt.title('Results of f(x) = 1 for n = %d (Deutsch-Jozsa algorithm)' % 4, fontsize=25)
        plt.xlabel('Measurement',fontsize=20)
        plt.ylabel('Frequency of occurence',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        
        fig.savefig('figures/dj_counts_histogram_%d.png' % 4, bbox_inches='tight')
    def benchmark_hack(self, max_bits, num_trials, npz_filename):
        provider = IBMQ.enable_account(API_TOKEN)
        backend = provider.backends.ibmq_16_melbourne

        run_time_arr = np.zeros([max_bits, num_trials])
        correct_arr = np.zeros([max_bits, num_trials])
        correct_percentage_arr = np.zeros([max_bits, num_trials])
        
        num_shots = 8192
        for n in range(1, max_bits+1):
            print("Starting benchmarking customs for n = %d" % (n))
            functions = [(CONSTANT, self.get_uf(n, "constant0")), (CONSTANT, self.get_uf(n, "constant1"))]
            for _ in range(3):
                functions.append( (BALANCED, self.get_uf(n, "balanced")) )

            for trial in range(num_trials):
                expected, uf = functions[trial]
                circuit = self.build_circuit(n, None, True, uf)
                counts, assembly_time, run_time = self.measure(circuit, backend, num_shots=num_shots)
                answer = self.evaluate(counts)
                correct = 100 if ((answer == '0'*n) ^ (expected == BALANCED)) else 0
                percentage = counts[answer] / float(num_shots) * 100
                print("Trial %d: expected %d. Ran in %f. correct: %d. percentage %f" % (trial, expected, run_time, correct, percentage))
                print(counts)

                run_time_arr[n-1, trial] = run_time / num_shots
                correct_arr[n-1, trial] = correct
                correct_percentage_arr[n-1, trial] = percentage
        np.savez(npz_filename, run_time_arr = run_time_arr, n_list = list(range(1, max_bits+1)), n = n, correct_arr = correct_arr)
    def plot_hack(self, npz_filename):
        data = np.load(npz_filename)

        run_time_arr = data['run_time_arr']
        n_list = data['n_list']

        avg_time = np.sum(run_time_arr,1)/np.shape(run_time_arr)[1]

        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        z = np.polyfit(n_list, avg_time, 10)
        p = np.poly1d(z)

        plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
        plt.plot(n_list, avg_time, ls = '', markersize = 15, marker = '.',label = 'M')
        plt.title('Execution time scaling for Deutsch-Jozsa algorithm (custom circuit)', fontsize=25)
        plt.xlabel('n (bit string length)',fontsize=20)
        plt.ylabel('Average time of execution (s)',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        fig.savefig('figures/dj_run_time_custom.png', bbox_inches='tight')

        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        correct_arr = data['correct_arr']
        n_list = data['n_list']

        avg_correct = np.sum(correct_arr,1)/np.shape(correct_arr)[1]
        
        plt.bar(n_list, avg_correct, align='center')#, tick_label=answers)
        plt.title('Percentage correct by majority. Deutsch-Jozsa algorithm (custom circuit)', fontsize=25)
        plt.xlabel('n (bit string length)',fontsize=20)
        plt.ylabel('Percentage Correct',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
            
        fig.savefig('figures/dj_correct_custom.png', bbox_inches='tight')

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
        provider = IBMQ.enable_account(API_TOKEN)
        backend = provider.backends.ibmq_london
        
        run_time_arr = np.zeros([max_bits, num_trials])
        assemble_time_arr = np.zeros([max_bits, num_trials])
        correct_arr = np.zeros([max_bits, num_trials])        

        num_shots = 8192
        for n in range(1, max_bits+1):
            print("Starting benchmarking for n = %d" % (n))
            functions = self.generate_functions(n, True, num_trials)
            trial = 0
            for trial in range(num_trials):
                expected, f = functions[trial]
                try:
                    counts, answer, assemble_time, run_time = self.run(f, backend, num_shots=num_shots)
                except ValueError:
                    print("Error")
                    continue
                print("Test finished for %s function. Got %s %d times." % (NAMES[expected], '0'*n, counts['0'*n]))
            
                run_time_arr[n-1, trial] = run_time / num_shots
                assemble_time_arr[n-1, trial] = assemble_time
                print(answer, '0'*n, answer == '0'*n)
                if expected == CONSTANT and answer == '0'*n:
                    correct_arr[n-1, trial] = 1.0
                    print("Marking right")
                elif expected == BALANCED and answer != '0'*n:
                    print("Marking right")
                    correct_arr[n-1, trial] = 1.0
                else:
                    print("marking wrong")
                    print("expected", expected, "answer", answer, counts['0'*n])

        num_times = 100
        variance_arr = np.zeros(num_times)
        n = 4
        print("Starting benchmarking for different Ufs for n = %d" % (n))
        functions = self.generate_functions(n, True, num_times)
        for trial in range(num_times):
            expected, f = functions[trial]
            try:
                counts, answer, assemble_time, run_time = self.run(f, backend, num_shots=num_shots)
            except:
                print("Error")
                variance_arr[trial] = 0.00325
                continue
            print("Test finished for %s function. Got %s %d times." % (NAMES[expected], '0'*n, counts['0'*n]))
            print(run_time / num_shots, f)
            
            variance_arr[trial] = run_time / num_shots
        np.savez(npz_filename, run_time_arr = run_time_arr, assemble_time_arr = assemble_time_arr, n_list = list(range(1, max_bits+1)), n = n, num_times = num_times, variance_arr = variance_arr, correct_arr = correct_arr)

    def plot_data(self, npz_filename):
        """
        Plot benchmarking data. Plot average run time for each n and variance data for 100 random Ufs.
        Args: 
        npz_filename: where tp load data
        """
        
        data = np.load(npz_filename)

        run_time_arr = data['run_time_arr']
        n_list = data['n_list']

        avg_time = np.sum(run_time_arr,1)/np.shape(run_time_arr)[1]

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

        fig.savefig('figures/dj_run_time.png', bbox_inches='tight')

        num_times = data['num_times']
        variance_arr = data['variance_arr']
        n = data['n']

        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        assemble_time_arr = data['assemble_time_arr']
        n_list = data['n_list']

        avg_time = np.sum(assemble_time_arr,1)/np.shape(assemble_time_arr)[1]

        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        z = np.polyfit(n_list, avg_time, 10)
        p = np.poly1d(z)

        plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
        plt.plot(n_list, avg_time, ls = '', markersize = 15, marker = '.',label = 'M')
        plt.title('Compilation time scaling for Deutsch-Jozsa algorithm', fontsize=25)
        plt.xlabel('n (bit string length)',fontsize=20)
        plt.ylabel('Average time of execution (s)',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        fig.savefig('figures/dj_assemble_time.png', bbox_inches='tight')

        correct_arr = data['correct_arr']
        n_list = data['n_list']

        avg_correct = np.sum(correct_arr,1)/np.shape(correct_arr)[1]

        plt.rcParams["font.family"] = "serif"
        fig = plt.figure(figsize=(16,10))

        z = np.polyfit(n_list, avg_correct, 10)
        p = np.poly1d(z)

        plt.plot(np.linspace(n_list[0], n_list[-1] + 0.1, 100), p(np.linspace(n_list[0], n_list[-1] + 0.1, 100)), ls = '-', color = 'r')
        plt.plot(n_list, avg_correct, ls = '', markersize = 15, marker = '.',label = 'M')
        plt.title('Percentage correct for Deutsch-Jozsa algorithm', fontsize=25)
        plt.xlabel('n (bit string length)',fontsize=20)
        plt.ylabel('Percentage correct',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        fig.savefig('figures/dj_correct.png', bbox_inches='tight')

        
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

    

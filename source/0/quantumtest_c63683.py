# https://github.com/grig-ar/Projects/blob/b36e8d45956d29ac5287e9571cc83ebf0b04fabe/4/Python/Quantum%20Tester/main/quantum/test/QuantumTest.py
from main.qiskit.tools.monitor import job_monitor

import qiskit as q
from main.quantum.circuit.GroverCircuit import GroverCircuit
from main.quantum.circuit.ShorCircuit import ShorCircuit


class QuantumTest(object):

    def __init__(self, test_name, test_circuit, test_shots, result):
        self.name = test_name
        self.circuit = test_circuit
        self.shots = test_shots
        self.result = result
        if test_circuit is not None:
            qubits_amount = test_circuit.num_qubits
            format_template = ':0={}b'.format(qubits_amount)
            format_template = "{%s}" % format_template
            self.binary_result = format_template.format(result)
        else:
            self.binary_result = ""
        self.results = []
        self.not_drawn_res = []

    def run_ibm(self, test_backend):
        job = q.execute(self.circuit, backend=test_backend, shots=self.shots, optimization_level=2)
        job_monitor(job)
        result = job.result()
        fields = ['Test_name', 'Result', 'Backend_name', 'Job_status', 'Date', 'Shots_taken', 'Time_taken',
                  'Accuracy', 'Counts']
        dict_data = dict.fromkeys(fields)
        dict_data['Test_name'] = self.name
        dict_data['Result'] = self.result
        dict_results = result.to_dict()
        dict_data['Backend_name'] = dict_results['backend_name']
        dict_data['Job_status'] = dict_results['status']
        dict_data['Date'] = dict_results['date']
        dict_data['Shots_taken'] = dict_results['results'][0]['shots']
        dict_data['Time_taken'] = round(float(dict_results['time_taken']), 3)
        dict_data['Accuracy'] = self.count_accuracy_binary(result.get_counts())
        dict_data['Counts'] = result.get_counts()
        self.results.append(dict_data)
        self.not_drawn_res.append(dict_data)

    def run_inspire(self, backend, qi):
        qasm_circuit = ""
        qubits_amount = 0
        if self.name == 'Shor_3_5':
            qasm_circuit = ShorCircuit.get_inspire_circuit(backend['name'])
            qubits_amount = 5
        elif self.name == 'Grover_2_bits':
            qubits_amount = 2
            qasm_circuit = GroverCircuit.get_inspire_circuit(qubits_amount, backend['name'])
        elif self.name == 'Grover_3_bits':
            qubits_amount = 3
            qasm_circuit = GroverCircuit.get_inspire_circuit(qubits_amount, backend['name'])
        elif self.name == 'Grover_4_bits':
            qubits_amount = 4
            qasm_circuit = GroverCircuit.get_inspire_circuit(qubits_amount, backend['name'])
        elif self.name == 'Grover_5_bits':
            qubits_amount = 5
            qasm_circuit = GroverCircuit.get_inspire_circuit(qubits_amount, backend['name'])

        format_template = ':0={}b'.format(qubits_amount)
        format_template = "{%s}" % format_template
        result = qi.execute_qasm(qasm_circuit, backend_type=backend, number_of_shots=self.shots)
        fields = ['Test_name', 'Result', 'Backend_name', 'Job_status', 'Date', 'Shots_taken', 'Time_taken',
                  'Accuracy', 'Counts']
        dict_data = dict.fromkeys(fields)
        dict_data['Test_name'] = self.name
        dict_data['Result'] = self.result
        dict_data['Backend_name'] = backend['name']
        dict_data['Job_status'] = 'OK'
        dict_data['Date'] = result['created_at']
        dict_data['Shots_taken'] = self.shots
        dict_data['Time_taken'] = round(float(result['execution_time_in_seconds']), 3)
        dict_data['Accuracy'] = self.count_accuracy_decimal(result['histogram'])
        raw_data = dict()
        for key in result['histogram'].keys():
            raw_data[format_template.format(int(key))] = round(result['histogram'][key] * self.shots)
        dict_data['Counts'] = raw_data
        self.results.append(dict_data)
        self.not_drawn_res.append(dict_data)
        # x = np.arange(2)
        # ticks_array = []
        # if result.get('histogram', {}):
        #     print(result['histogram'])
        #     plt.bar(x, height=result['histogram'].values())
        #     # plt.xticks(x, result['histogram'].keys())
        #     for key in result['histogram'].keys():
        #         ticks_array.append("{:0=3b}".format(int(key)))
        #     plt.xticks(x, ticks_array)
        #     plt.show()

    def count_accuracy_binary(self, counts):
        for key in counts:
            if key == self.binary_result:
                return round(counts[key] / self.shots, 2)
        return 0

    def count_accuracy_decimal(self, counts):
        for key in counts.keys():
            if key == str(self.result):
                return round(counts[key], 2)
        return 0

    def get_data(self, index):
        return self.results[index]['Counts']

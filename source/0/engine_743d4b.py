# https://github.com/brunopacheco1/simple-rules-engine/blob/0e8eba99952fe77c93753599c634b9e86998386f/engine.py
from qiskit import Aer, QuantumCircuit, execute
from qiskit.quantum_info import Statevector
from input import parse_rule_input, RuleInput
from output import rules_output
from circuit import engine_circuit

simulator = Aer.get_backend('qasm_simulator')

def hit_rules(ruleInput: RuleInput):
    input = parse_rule_input(ruleInput)
    rules_circuit = QuantumCircuit(9,5)
    rules_circuit.initialize(Statevector.from_label(input), range(4))
    rules_circuit.append(engine_circuit, range(9))
    rules_circuit.measure([4,5,6,7,8],[0,1,2,3,4])
    job = execute(rules_circuit, backend=simulator, shots=1)
    result = job.result()
    counts = result.get_counts()
    return rules_output[list(counts.keys())[0]]

# https://github.com/TobiasRohner/ApproximateReversibleCircuits/blob/0ea8270af169da804593a357954829c89fc0b9e8/errors.py
import qiskit
import copy
import qc_properties



def compute_error_rates(qc, function, noise_model=qc_properties.noise_model):
    e = 0
    fn = 0
    fp = 0
    jobs = []
    backend = qiskit.Aer.get_backend('qasm_simulator')
    shots = 1024
    for inp in range(2**function.input_size):
        initialized_qc = copy.deepcopy(qc)
        data = initialized_qc.data
        initializing = [(qiskit.extensions.XGate(), [initialized_qc.qubits[i]], []) for i in range(function.input_size) if (inp >> i) & 1 == 1]
        data = initializing + data
        initialized_qc.data = data
        jobs.append(qiskit.execute(initialized_qc,
                                   backend,
                                   basis_gates=qc_properties.basis_gates,
                                   coupling_map=qc_properties.coupling_map,
                                   shots=shots,
                                   noise_model=noise_model))
    simulated_results = [job.result().get_counts() for job in jobs]
    exact_results = [function(inp) for inp in range(2**function.input_size)]
    num_positive = sum([bin(ex).count('1') for ex in exact_results])
    for simulated, exact in zip(simulated_results, exact_results):
        for bit in range(function.output_size):
            correct_cnt = sum([v if k.rjust(function.output_size, '0')[bit]==bin(exact)[2:].rjust(function.output_size, '0')[bit] else 0 for k,v in simulated.items()])
            e += shots - correct_cnt
            if bin(exact)[2:].rjust(function.output_size, '0')[bit] == 0:
                fp += shots - correct_cnt
            else:
                fn += shots - correct_cnt
    e /= shots * 2**function.input_size * function.output_size
    fn /= shots * num_positive
    fp /= shots * (2**function.input_size - num_positive)
    return e, fn, fp



def reduce_noise(noise_model, factor):
    def reduce_probs(probs, f, idx):
        acc = 0
        for i in range(len(probs)):
            if i == idx:
                continue
            acc += (1-1/f) * probs[i]
            probs[i] /= f
        probs[idx] += acc
        return probs

    d = noise_model.to_dict()
    for err in d['errors']:
        if err['type'] == 'qerror':
            err['probabilities'] = reduce_probs(err['probabilities'], factor, len(err['probabilities'])-1)
        elif err['type'] == 'roerror':
            err['probabilities'] = [reduce_probs(probs, factor, i) for i, probs in enumerate(err['probabilities'])]
        else:
            raise RuntimeError('Unknown error type')
    return qiskit.providers.aer.noise.NoiseModel.from_dict(d)

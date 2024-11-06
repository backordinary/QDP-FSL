# https://github.com/Sav0s/Quantum-RSA-Decrypter/blob/729086608e097997b1774a134b4c0963c764d315/Quantum-RSA-Decryptor/Decrypt.py
import qsharp

from IntegerFactorization import FactorInteger
from qiskit import IBMQ, BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Shor
from qiskit.providers.ibmq import least_busy
from qiskit.test.providers import provider
from qsharp.clients.iqsharp import IQSharpError
from qiskit.tools.monitor import job_monitor
from util import printProgress

from account import token


class Decryptor:
    def factorize(self, factor):
        pass


class NumericDecryptor(Decryptor):
    def __init__(self, factor):
        self.factor = factor

    def factorize(self):
        result = list()
        for i in range(2, self.factor):
            if self.factor % i == 0:
                result.append(i)
                secondResult = int(self.factor / i)
                result.append(secondResult)
                break
            
            # Visualisierung der noch zu testenden Faktoren
            if i % 1000 == 2:
                printProgress(i, self.factor)
        print("\n")
        return result[0], result[1]


class IBMDecryptor(Decryptor):
    def __init__(self, factor):
        self.factor = factor

    def factorize(self):
        shor = Shor(self.factor)

        # If you use get_backend('qasm_simulator') don't factor numbers greater than 15, it lasts nearly forever
        backend = BasicAer.get_backend('qasm_simulator')
        print(f"Using backend: {backend}")
        quantum_instance = QuantumInstance(backend, shots=1)
        computation = shor.run(quantum_instance)
        if len(computation['factors']) == 0:
            print("Algorithm went wrong")
            return None, None
        result = computation['factors'][0]
        return result[0], result[1]


class IBMDecryptorReal(Decryptor):
    def __init__(self, factor):
        self.factor = factor
        storedAccount = IBMQ.stored_account()
        if storedAccount == None or storedAccount == {}:
            IBMQ.save_account(token)
        self.__provider = IBMQ.load_account()

    def factorize(self):
        shor = Shor(self.factor)

        device = least_busy(self.__provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and
                                                                not x.configuration().simulator and x.status().operational == True))
        print("Running on current least busy device: ", device)

        quantum_instance = QuantumInstance(device, shots=1024, skip_qobj_validation=False)
        computation = shor.run(quantum_instance)
        if len(computation['factors']) == 0:
            print("Algorithm went wrong")
            return None, None
        result = computation['factors'][0]
        return result[0], result[1]


class QSharpDecryptor(Decryptor):
    def __init__(self, factor):
        self.factor = factor

    def factorize(self):
        output = FactorInteger.simulate(
            number=self.factor,
            useRobustPhaseEstimation=True)
        if output == None:
            print("The computation was too big for Q#")
            return None, None
        return output


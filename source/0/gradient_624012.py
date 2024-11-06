# https://github.com/SchmidtMoritz/IAGQ/blob/476610f4975247e99ecf0ff3ffcbbe6dafcb901c/src/IAGQ/gradient.py
import abc

from .circuit_runner import CircuitRunner
import numpy as np
from abc import ABC
from qiskit.opflow import (
    MatrixOp,
    StateFn,
    CircuitStateFn,
    CircuitSampler,
)
from qiskit.opflow.gradients import Gradient as GD
from qiskit.utils import QuantumInstance
from qiskit import Aer
import time
from .ansatz import Ansatz
from .observable import Observable


class Gradient(ABC):
    """
    abstract gradient base class
    """

    def __init__(
        self,
        ansatz: Ansatz,
        observable: Observable,
        backend: str,
        label: str = "",
        plot_label: str = None,
    ):
        """
        :param label: label used in mlflow. not supporting all special characters
        :param plot_label: label used in plots. also allowing special notation like  "=" or "$\gamma$"
        """
        self.label = label
        if plot_label is None:
            self.plot_label = label
        else:
            self.plot_label = plot_label
        self.ansatz = ansatz
        self.observable = observable
        self.backend = backend

    @abc.abstractmethod
    def evaluate_gradient(
        self, initial_point: np.ndarray, parameter_indices: list[int], shots: int
    ) -> np.ndarray:
        """
        interface to compute gradient approximation

        :param initial_point: parameter point at which to compute gradient
        :param parameter_indices: list of parameter indices to compute gradients for
        :param shots: amount of shots used for each individual measurement
        :return: always returns gradient array of size (k,).
                 entries at indices that are not part of parameter_indices are filled with NaNs
        """
        return NotImplemented

    @abc.abstractmethod
    def evaluate_gradient_and_sample_variance(
        self, initial_point: np.ndarray, parameter_indices: list[int], shots: int
    ) -> np.ndarray:
        """
        interface to compute gradient approximation and also return sample variance

        :param initial_point: parameter point at which to compute gradient
        :param parameter_indices: list of parameter indices to compute gradients for
        :param shots: amount of shots used for each individual measurement
        :return: always returns gradient array of size (k,) and sample variance array of size (k,).
                 entries at indices that are not part of parameter_indices are filled with NaNs
        """
        return NotImplemented

    @abc.abstractmethod
    def evaluate_gradient_variance(
        self, initial_point: np.ndarray, parameter_indices: list[int], shots: int
    ) -> np.ndarray:
        """
        interface to compute gradient variance

        :param initial_point: parameter point at which to compute gradient variance
        :param parameter_indices: list of parameter indices to compute gradient variances for
        :param shots: amount of shots used for each individual measurement
                      -normally used with opflow to compute exact gradient variance, then shots do not matter
                      -but could also be used to approximately measure gradient variance using shots
        :return: always returns gradient variance array of size (k,).
                 entries at indices that are not part of parameter_indices are filled with NaNs
        """
        return NotImplemented

    def evaluate_average_measurement_variance(
        self, initial_point: np.ndarray, parameter_indices: list[int], shots: int
    ) -> np.ndarray:
        """
        interface to compute average measurement variance for doubly stochastical gradients e.g. spsa

        there you want to be able to differentiate between variance for a current choice of a random variable
        and the overall variance

        :param initial_point: parameter point
        :param parameter_indices: list of parameter indices
        :param shots: amount of shots used for each individual measurement
        :return: always returns average measurement variance array of size (k,).
                 entries at indices that are not part of parameter_indices are filled with NaNs

                 default case: just return variance (e.g. NOT doubly stochastical)
        """

        return self.evaluate_gradient_variance(initial_point, parameter_indices, shots)

    def evaluate_variance_of_expectations(
        self, initial_point: np.ndarray, parameter_indices: list[int], shots: int
    ) -> np.ndarray:
        """
        interface to compute variance of expectations for doubly stochastical gradients e.g. spsa

        there you want to be able to differentiate between variance for a current choice of a random variable
        and the overall variance

        :param initial_point: parameter point
        :param parameter_indices: list of parameter indices
        :param shots: amount of shots used for each individual measurement
        :return: always returns variance of expectations array of size (k,).
                 entries at indices that are not part of parameter_indices are filled with NaNs

                 default case: just return zero array (e.g. NOT doubly stochastical)
        """
        return np.zeros((self.ansatz.get_parameter_count(),))

    @abc.abstractmethod
    def get_metadata(self):
        """return dictionary with meta data to be displayed in mlflow"""
        meta_dict = {}
        meta_dict.update(self.ansatz.get_metadata())
        meta_dict.update(self.observable.get_metadata())
        meta_dict["Backend"] = self.backend

        return meta_dict


class ParameterShift(Gradient):
    """
    classical Parameter Shift (arXiv:1811.11184)

    usable on gates with two eigenvalues +/- r

    g_i = r * (L(theta + pi /(4r)) - L(theta - pi /(4r)))
    """

    def __init__(self, ansatz, observable, backend, label="PS", plot_label=None):
        super().__init__(ansatz, observable, backend, label, plot_label)
        assert self.ansatz.is_parameter_shiftable()
        self.circuitrunner = CircuitRunner(ansatz, observable, backend)

    def evaluate_gradient(self, initial_point, parameter_indices, shots):
        r = self.ansatz.get_r_values()
        gradient = np.empty((len(r),), dtype=np.float)
        gradient[:] = np.nan

        for i in parameter_indices:
            shift = np.zeros((len(r),))
            shift[i] += np.pi / (4 * r[i])

            plus_shift_value = self.circuitrunner.run(initial_point + shift, shots)
            minus_shift_value = self.circuitrunner.run(initial_point - shift, shots)

            gradient[i] = r[i] * (plus_shift_value - minus_shift_value)

        return gradient

    def evaluate_gradient_and_sample_variance(
        self, initial_point, parameter_indices, shots
    ):
        r = self.ansatz.get_r_values()
        gradient = np.empty((len(r),), dtype=np.float)
        gradient[:] = np.nan

        variance = np.empty((len(r),), dtype=np.float)
        variance[:] = np.nan

        for i in parameter_indices:
            shift = np.zeros((len(r),))
            shift[i] += np.pi / (4 * r[i])

            plus_shift_value, plus_variance = self.circuitrunner.run(
                initial_point + shift, shots, comp_variance=True
            )
            minus_shift_value, minus_variance = self.circuitrunner.run(
                initial_point - shift, shots, comp_variance=True
            )
            gradient[i] = r[i] * (plus_shift_value - minus_shift_value)

            variance[i] = r[i] ** 2 * (plus_variance + minus_variance)

        return gradient, variance

    def evaluate_gradient_variance(self, initial_point, parameter_indices, shots):

        r = self.ansatz.get_r_values()

        variance = np.empty((len(r),), dtype=np.float)
        variance[:] = np.nan

        for i in parameter_indices:
            shift = np.zeros((len(r),))
            shift[i] += np.pi / (4 * r[i])

            plus_shift_value = self.circuitrunner.run(initial_point + shift, shots)
            minus_shift_value = self.circuitrunner.run(initial_point - shift, shots)

            plus_squared_shift_value = self.circuitrunner.run(
                initial_point + shift, shots, squared=True
            )
            minus_squared_shift_value = self.circuitrunner.run(
                initial_point - shift, shots, squared=True
            )

            var_plus = plus_squared_shift_value - (plus_shift_value**2)
            var_minus = minus_squared_shift_value - (minus_shift_value**2)

            variance[i] = (var_plus + var_minus) * r[i] ** 2

        return variance

    def get_metadata(self):
        meta_dict = super().get_metadata()
        meta_dict["Gradient"] = "Parameter-Shift"

        return meta_dict


class FiniteDifferenzen(Gradient):
    """
    :param h_list: list of shift values h

    coefficients get computed accordingly
    """

    def __init__(
        self,
        ansatz,
        observable,
        h_list,
        backend,
        label="FD",
        plot_label=None,
    ):
        super().__init__(ansatz, observable, backend, label, plot_label)

        assert len(h_list) == len(set(h_list))  # only unique shift values

        self.circuitrunner = CircuitRunner(ansatz, observable, backend)
        self.h_list = h_list
        self.coefficient_list = self.calculate_coefficients(h_list)

    @staticmethod
    def calculate_coefficients(h_list):

        N = len(h_list)
        assert N >= 2

        h_array = np.array(h_list)

        A = np.ones((N, N))
        for i in range(1, N):
            A[i, :] = h_array / i * A[i - 1, :]

        b = np.zeros((N,))
        b[1] = 1

        coefficients = np.linalg.solve(A, b)
        return coefficients

    def set_h_list(self, h_list):

        assert len(h_list) == len(set(h_list))

        self.h_list = h_list
        self.coefficient_list = self.calculate_coefficients(h_list)

    def evaluate_gradient(self, initial_point, parameter_indices, shots):
        p = self.ansatz.get_parameter_count()
        gradient = np.empty((p,), dtype=np.float)
        gradient[:] = np.nan

        for i in parameter_indices:
            shift = np.zeros((p,))
            grad_value = 0

            for j in range(len(self.h_list)):

                shift[i] = self.h_list[j]
                grad = self.circuitrunner.run(initial_point + shift, shots)

                grad_value += self.coefficient_list[j] * grad

            gradient[i] = grad_value

        return gradient

    def evaluate_gradient_and_sample_variance(
        self, initial_point, parameter_indices, shots
    ):
        p = self.ansatz.get_parameter_count()
        gradient = np.empty((p,), dtype=np.float)
        gradient[:] = np.nan

        variance = np.empty((p,), dtype=np.float)
        variance[:] = np.nan

        for i in parameter_indices:
            shift = np.zeros((p,))
            grad_value = 0
            var_value = 0

            for j in range(len(self.h_list)):

                shift[i] = self.h_list[j]

                grad, var = self.circuitrunner.run(
                    initial_point + shift, shots, comp_variance=True
                )

                var_value += (self.coefficient_list[j] ** 2) * var
                grad_value += self.coefficient_list[j] * grad

            gradient[i] = grad_value
            variance[i] = var_value

        return gradient, variance

    def evaluate_gradient_variance(self, initial_point, parameter_indices, shots):
        p = self.ansatz.get_parameter_count()

        variance = np.empty((p,), dtype=np.float)
        variance[:] = np.nan

        for i in parameter_indices:

            shift = np.zeros((p,))
            grad_value = 0
            var_value = 0

            for j in range(len(self.h_list)):
                shift[i] = self.h_list[j]

                shift_val = self.circuitrunner.run(initial_point + shift, shots)
                shift_val_squared = self.circuitrunner.run(
                    initial_point + shift, shots, squared=True
                )

                var_value += (self.coefficient_list[j] ** 2) * (
                    shift_val_squared - (shift_val**2)
                )

            variance[i] = var_value

        return variance

    def get_metadata(self):

        meta_dict = super().get_metadata()

        meta_dict["Gradient"] = "Finite-Differences"
        meta_dict["Samplepoint-Distances"] = str(self.h_list)

        return meta_dict


class QGFGradient(Gradient):
    def __init__(
        self,
        ansatz,
        observable,
        backend,
        grad_method,
        label="QGF",
        plot_label=None,
    ):
        """
        Wrapper class for qiskit gradient framework

        https://github.com/Qiskit/qiskit-terra/tree/main/qiskit/opflow/gradients
        https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/operators/02_gradients_framework.ipynb

        :param grad_method:  possible values: param_shift, fin_diff, lin_comb
        """
        super().__init__(ansatz, observable, backend, label, plot_label)

        self.backend = backend
        self.grad_method = grad_method

    def evaluate_gradient(self, initial_point, parameter_indices, shots):

        gradient = np.empty((self.ansatz.get_parameter_count(),), dtype=np.float)
        gradient[:] = np.nan

        circuit = self.ansatz.circuit
        circuit_state_fn = CircuitStateFn(circuit)

        observable_state_fn = StateFn(
            MatrixOp(self.observable.get_matrix()), is_measurement=True
        )
        op = observable_state_fn @ circuit_state_fn

        state_grad = GD(grad_method=self.grad_method).convert(
            operator=op, params=self.ansatz.get_parameters()
        )
        value_dict = {
            self.ansatz.get_parameters()[i]: initial_point[i]
            for i in range(self.ansatz.get_parameter_count())
        }

        if self.backend == "simulator":

            q_instance = QuantumInstance(
                Aer.get_backend("qasm_simulator"), shots=shots, seed_simulator=40
            )

            sampler = CircuitSampler(q_instance, attach_results=True)
            expectation_operator = state_grad.assign_parameters(
                value_dict
            )  # matches Expectation().convert
            samples = sampler.convert(expectation_operator)

            state_grad_result = samples.eval()

        else:
            state_grad_result = state_grad.assign_parameters(value_dict).eval()

        for i in parameter_indices:
            gradient[i] = state_grad_result[i].real

        return gradient

    def evaluate_gradient_and_sample_variance(
        self, initial_point, parameter_indices, shots
    ):
        return NotImplemented

    def evaluate_gradient_variance(self, initial_point, parameter_indices, shots):
        return NotImplemented

    def get_metadata(self):

        meta_dict = super().get_metadata()

        meta_dict["Gradient"] = "Qiskit Gradient Framework"
        meta_dict["QFG Gradient Method"] = self.grad_method
        if self.grad_method == "fin_diff":
            meta_dict["Epsilon"] = str(10**-6)

        return meta_dict


class SPSAGradient(Gradient):
    """
    SPSA Gradient

    :param h: shift value
    :param exact_variance: flag if variances using delta should be approximated or not

    Delta vector:  in each dimension +1 or -1 with prob 0.5
    """

    def __init__(
        self,
        ansatz,
        observable,
        h,
        backend,
        label="SPSA",
        plot_label=None,
    ):
        super().__init__(ansatz, observable, backend, label, plot_label)
        self.circuitrunner = CircuitRunner(ansatz, observable, backend)
        self.h = h
        self.exact_variance = False
        np.random.seed(40)

    def evaluate_gradient(self, initial_point: np.ndarray, parameter_indices, shots):

        delta = np.random.binomial(n=1, p=0.5, size=self.ansatz.get_parameter_count())
        delta = np.ones((self.ansatz.get_parameter_count())) - (2 * delta)

        plus_shift = self.circuitrunner.run(initial_point + self.h * delta, shots)
        minus_shift = self.circuitrunner.run(initial_point - self.h * delta, shots)

        gradient = (plus_shift - minus_shift) / (2 * self.h) * delta

        return gradient

    def evaluate_gradient_and_sample_variance(
        self, initial_point, parameter_indices, shots
    ):

        delta = np.random.binomial(n=1, p=0.5, size=self.ansatz.get_parameter_count())
        delta = np.ones((self.ansatz.get_parameter_count())) - (2 * delta)

        plus_shift, plus_var = self.circuitrunner.run(
            initial_point + self.h * delta, shots, comp_variance=True
        )
        minus_shift, minus_var = self.circuitrunner.run(
            initial_point - self.h * delta, shots, comp_variance=True
        )

        gradient = (plus_shift - minus_shift) / (2 * self.h) * delta
        variance = (plus_var + minus_var) / np.power((2 * self.h) * delta, 2)

        return gradient, variance

    def evaluate_gradient_variance(self, initial_point, parameter_indices, shots):

        average_measurement_variance = self.evaluate_average_measurement_variance(
            initial_point, parameter_indices, shots
        )
        variance_of_expectations = self.evaluate_variance_of_expectations(
            initial_point, parameter_indices, shots
        )
        print(average_measurement_variance)
        print(variance_of_expectations)

        return average_measurement_variance + variance_of_expectations

    def gen_delta_from_i(self, i: int) -> np.ndarray:
        """
        Transforms number i to a deltavector for SPSA

        transform number to bitstring, then map 0->1 and 1->-1

        used to iterate over all possible delta vectors
        """

        n = self.ansatz.get_parameter_count()
        bitstring = np.binary_repr(i, width=n)
        delta = np.ones((n,))
        for j in range(n):

            if bitstring[j] == "1":
                delta[j] *= -1

        return delta

    def evaluate_average_measurement_variance(
        self, initial_point, parameter_indices, shots
    ):
        """
        \sigma^2_M in thesis

        if exact:
        iterates over first half of all possible delta vectors and
        computes running average of measurement variances along the way

        can have long runtime: prints out estimated remaining runtime when used
        """

        if self.exact_variance:
            variance = np.zeros((self.ansatz.get_parameter_count(),))
            old_variance = variance
            time_per_run = 0
            time_interval = 20
            start_time = time.time()

            for i in range(0, 2 ** (self.ansatz.get_parameter_count() - 1)):

                if i % (time_interval - 1) == 0:
                    stop_time = time.time()
                    time_per_run = (stop_time - start_time) / time_interval
                    start_time = time.time()

                delta = self.gen_delta_from_i(i)

                plus_shift = self.circuitrunner.run(
                    initial_point + self.h * delta, shots
                )
                minus_shift = self.circuitrunner.run(
                    initial_point - self.h * delta, shots
                )

                plus_squared = self.circuitrunner.run(
                    initial_point + self.h * delta, shots, squared=True
                )
                minus_squared = self.circuitrunner.run(
                    initial_point - self.h * delta, shots, squared=True
                )

                plus_variance = plus_squared - (plus_shift**2)
                minus_variance = minus_squared - (minus_shift**2)

                variance_i = (plus_variance + minus_variance) / np.power(
                    (2 * self.h) * delta, 2
                )
                variance = (variance_i + i * variance) / (i + 1)

                diff = np.linalg.norm(old_variance - variance)
                old_variance = variance

                if i % 100 == 0:
                    est_rem_runtime = time_per_run * (
                        2 ** (self.ansatz.get_parameter_count() - 1) - (i + 1)
                    )
                    print(i, delta, variance, diff)
                    print(
                        "Estimated remaining runtime: ",
                        est_rem_runtime // 3600,
                        "h, ",
                        (est_rem_runtime // 60) % 60,
                        "m, ",
                        est_rem_runtime % 60,
                        "s, ",
                    )
            return variance
        else:
            return self.approximate_average_measurement_variance(
                initial_point, parameter_indices, shots
            )

    def approximate_average_measurement_variance(
        self, initial_point, parameter_indices, shots
    ):
        """
        \sigma^2_M in thesis

        approximates average measurement variance by sampling random delta vectors
        if over the last 20 samples the average hasnt changed more then epsilon (mean over all parameters)
            ->  accept approximation

        parameter_indices is not used since spsa is approximating for all parameters anyway
        """

        variance = np.zeros((self.ansatz.get_parameter_count(),))
        count = 0
        epsilon = 10**-2
        old_variance = variance
        min_count = 20
        convergence_interval = 20
        last_diffrences = np.zeros((convergence_interval,))

        while (
            np.mean(last_diffrences) > epsilon * self.ansatz.get_parameter_count()
            or count < min_count
        ):

            delta = np.random.binomial(
                n=1, p=0.5, size=self.ansatz.get_parameter_count()
            )
            delta = np.ones((self.ansatz.get_parameter_count())) - (2 * delta)

            plus_shift = self.circuitrunner.run(initial_point + self.h * delta, shots)
            minus_shift = self.circuitrunner.run(initial_point - self.h * delta, shots)

            plus_squared = self.circuitrunner.run(
                initial_point + self.h * delta, shots, squared=True
            )
            minus_squared = self.circuitrunner.run(
                initial_point - self.h * delta, shots, squared=True
            )

            plus_variance = plus_squared - (plus_shift**2)
            minus_variance = minus_squared - (minus_shift**2)

            variance_i = (plus_variance + minus_variance) / np.power(
                (2 * self.h) * delta, 2
            )
            variance = (variance_i + count * old_variance) / (count + 1)
            count += 1

            diff = np.linalg.norm(old_variance - variance)
            last_diffrences[count % convergence_interval] = diff

            old_variance = variance
            """
            if count % 100 == 0:
                print(count, delta, np.mean(last_diffrences))
            """
        return variance

    def evaluate_gradient_expectation(self, initial_point, parameter_indices, shots):
        """
        evaluate E[g_i] independent of \Delta

        if exact:
        iterates over first half of all possible delta vectors and
        computes running average of measurement variances along the way

        can have long runtime: prints out estimated remaining runtime when used

        parameter_indices is not used since spsa is approximating for all parameters anyway
        """
        gradient_expectation = np.zeros((self.ansatz.get_parameter_count(),))
        old_gradient_expectation = gradient_expectation
        time_per_run = 0
        time_interval = 20
        start_time = time.time()

        for i in range(0, 2 ** (self.ansatz.get_parameter_count() - 1)):

            if i % (time_interval - 1) == 0:
                stop_time = time.time()
                time_per_run = (stop_time - start_time) / time_interval
                start_time = time.time()

            delta = self.gen_delta_from_i(i)

            plus_shift = self.circuitrunner.run(initial_point + self.h * delta, shots)
            minus_shift = self.circuitrunner.run(initial_point - self.h * delta, shots)

            gradient_expectation_i = (plus_shift - minus_shift) / ((2 * self.h) * delta)

            gradient_expectation = (
                gradient_expectation_i + i * gradient_expectation
            ) / (i + 1)

            diff = np.linalg.norm(old_gradient_expectation - gradient_expectation)
            old_gradient_expectation = gradient_expectation

            if i % 100 == 0:
                est_rem_runtime = time_per_run * (
                    2 ** (self.ansatz.get_parameter_count() - 1) - (i + 1)
                )
                print(i, delta, gradient_expectation, diff)
                print(
                    "Estimated remaining runtime: ",
                    est_rem_runtime // 3600,
                    "h, ",
                    (est_rem_runtime // 60) % 60,
                    "m, ",
                    est_rem_runtime % 60,
                    "s, ",
                )
        return gradient_expectation

    def approximate_gradient_expectation(self, initial_point, parameter_indices, shots):
        """
        approximate E[g_i] independent of \Delta

        approximates E[g_i] by sampling random delta vectors
        if over the last 20 samples the average hasnt changed more then epsilon (mean over all parameters)
            ->  accept approximation

        parameter_indices is not used since spsa is approximating for all parameters anyway
        """
        gradient_expectation = np.zeros((self.ansatz.get_parameter_count(),))
        count = 0
        epsilon = 10**-3
        old_gradient_expectation = gradient_expectation
        min_count = 20
        convergence_interval = 20
        last_diffrences = np.zeros((convergence_interval,))

        while (
            np.mean(last_diffrences) > epsilon * self.ansatz.get_parameter_count()
            or count < min_count
        ):

            delta = np.random.binomial(
                n=1, p=0.5, size=self.ansatz.get_parameter_count()
            )
            delta = np.ones((self.ansatz.get_parameter_count())) - (2 * delta)

            plus_shift = self.circuitrunner.run(initial_point + self.h * delta, shots)
            minus_shift = self.circuitrunner.run(initial_point - self.h * delta, shots)

            gradient_expectation_i = (plus_shift - minus_shift) / ((2 * self.h) * delta)

            gradient_expectation = (
                gradient_expectation_i + count * old_gradient_expectation
            ) / (count + 1)
            count += 1

            diff = np.linalg.norm(old_gradient_expectation - gradient_expectation)

            last_diffrences[count % convergence_interval] = diff

            old_gradient_expectation = gradient_expectation
            """
            if count % 100 == 0:
                print(count, delta, np.mean(last_diffrences))
            """
        return gradient_expectation

    def evaluate_variance_of_expectations(
        self, initial_point, parameter_indices, shots
    ):
        """
        evaluate \sigma^2_\Delta

        if exact:
        iterates over first half of all possible delta vectors and
        computes running average of variances of expectations along the way

        can have long runtime: prints out estimated remaining runtime when used
        """
        if self.exact_variance:
            exact_grad_expec = self.evaluate_gradient_expectation(
                initial_point, parameter_indices, shots
            )

            variance = np.zeros((self.ansatz.get_parameter_count(),))
            old_variance = variance
            time_per_run = 0
            time_interval = 20
            start_time = time.time()

            for i in range(0, 2 ** (self.ansatz.get_parameter_count() - 1)):

                if i % (time_interval - 1) == 0:
                    stop_time = time.time()
                    time_per_run = (stop_time - start_time) / time_interval
                    start_time = time.time()

                delta = self.gen_delta_from_i(i)

                plus_shift = self.circuitrunner.run(
                    initial_point + self.h * delta, shots
                )
                minus_shift = self.circuitrunner.run(
                    initial_point - self.h * delta, shots
                )

                x_i = (plus_shift - minus_shift) / ((2 * self.h) * delta)
                variance_i = np.power(x_i - exact_grad_expec, 2)

                variance = (variance_i + i * variance) / (i + 1)

                diff = np.linalg.norm(old_variance - variance)
                old_variance = variance

                if i % 100 == 0:
                    est_rem_runtime = time_per_run * (
                        (2 ** (self.ansatz.get_parameter_count() - 1)) - (i + 1)
                    )
                    print(i, delta, variance, diff)
                    print(
                        "Estimated remaining runtime: ",
                        est_rem_runtime // 3600,
                        "h, ",
                        (est_rem_runtime // 60) % 60,
                        "m, ",
                        est_rem_runtime % 60,
                        "s, ",
                    )

            return variance

        else:
            return self.approximate_variance_of_expectations(
                initial_point, parameter_indices, shots
            )

    def approximate_variance_of_expectations(
        self, initial_point, parameter_indices, shots
    ):
        """
        approximate \sigma^2_\Delta

        approximates \sigma^2_\Delta by sampling random delta vectors
        if over the last 20 samples the average hasnt changed more then epsilon (mean over all parameters)
            ->  accept approximation

        """
        approx_grad_expec = self.approximate_gradient_expectation(
            initial_point, parameter_indices, shots
        )

        variance = np.zeros((self.ansatz.get_parameter_count(),))
        count = 0
        epsilon = 10**-2
        old_variance = variance
        min_count = 20
        convergence_interval = 20
        last_diffrences = np.zeros((convergence_interval,))

        while (
            np.mean(last_diffrences) > epsilon * self.ansatz.get_parameter_count()
            or count < min_count
        ):
            delta = np.random.binomial(
                n=1, p=0.5, size=self.ansatz.get_parameter_count()
            )
            delta = np.ones((self.ansatz.get_parameter_count())) - (2 * delta)

            plus_shift = self.circuitrunner.run(initial_point + self.h * delta, shots)
            minus_shift = self.circuitrunner.run(initial_point - self.h * delta, shots)

            x_i = (plus_shift - minus_shift) / ((2 * self.h) * delta)
            variance_i = np.power(x_i - approx_grad_expec, 2)

            variance = (variance_i + count * variance) / (count + 1)

            count += 1

            diff = np.linalg.norm(old_variance - variance)

            last_diffrences[count % convergence_interval] = diff

            old_variance = variance
            """
            if count % 100 == 0:
                print(count, delta, np.mean(last_diffrences))
            """
        return variance

    def get_metadata(self):
        meta_dict = super().get_metadata()

        meta_dict["Gradient"] = "SPSA"
        meta_dict["Epsilon"] = str(self.h)

        return meta_dict


class GeneralParameterShift(Gradient):
    """
    general Parameter Shift (arXiv:2106.01388)

    usable on gates with two eigenvalues +/- r

    g_i = r/sin(2r gamma) * (L(theta + gamma) - L(\theta - gamma))
    """

    def __init__(
        self,
        ansatz,
        observable,
        gamma,
        backend,
        label="GPS",
        plot_label=None,
    ):
        super().__init__(ansatz, observable, backend, label, plot_label)

        assert self.ansatz.is_parameter_shiftable()
        self.circuitrunner = CircuitRunner(ansatz, observable, backend)
        self.gamma = gamma

    def evaluate_gradient(
        self, initial_point, parameter_indices, shots, comp_variance=False
    ):

        r = self.ansatz.get_r_values()
        gradient = np.empty((len(r),), dtype=np.float)
        gradient[:] = np.nan

        for i in parameter_indices:
            shift = np.zeros((len(r),))
            shift[i] += self.gamma

            plus_shift_value = self.circuitrunner.run(initial_point + shift, shots)
            minus_shift_value = self.circuitrunner.run(initial_point - shift, shots)

            gradient[i] = (
                r[i]
                / np.sin(2 * r[i] * self.gamma)
                * (plus_shift_value - minus_shift_value)
            )

        return gradient

    def evaluate_gradient_and_sample_variance(
        self, initial_point, parameter_indices, shots
    ):

        r = self.ansatz.get_r_values()

        gradient = np.empty((len(r),), dtype=np.float)
        gradient[:] = np.nan

        variance = np.empty((len(r),), dtype=np.float)
        variance[:] = np.nan

        for i in parameter_indices:
            shift = np.zeros((len(r),))
            shift[i] += self.gamma

            plus_shift_value, plus_variance = self.circuitrunner.run(
                initial_point + shift, shots, comp_variance=True
            )
            minus_shift_value, minus_variance = self.circuitrunner.run(
                initial_point - shift, shots, comp_variance=True
            )

            gradient[i] = (
                r[i]
                / np.sin(2 * r[i] * self.gamma)
                * (plus_shift_value - minus_shift_value)
            )

            variance[i] = ((r[i] / np.sin(2 * r[i] * self.gamma)) ** 2) * (
                plus_variance + minus_variance
            )

        return gradient, variance

    def evaluate_gradient_variance(self, initial_point, parameter_indices, shots):

        r = self.ansatz.get_r_values()
        variance = np.empty((len(r),), dtype=np.float)
        variance[:] = np.nan

        for i in parameter_indices:
            shift = np.zeros((len(r),))
            shift[i] += self.gamma

            plus_shift_value = self.circuitrunner.run(initial_point + shift, shots)
            minus_shift_value = self.circuitrunner.run(initial_point - shift, shots)

            plus_shift_squared = self.circuitrunner.run(
                initial_point + shift, shots, squared=True
            )
            minus_shift_squared = self.circuitrunner.run(
                initial_point - shift, shots, squared=True
            )

            plus_variance = plus_shift_squared - (plus_shift_value**2)
            minus_variance = minus_shift_squared - (minus_shift_value**2)

            variance[i] = ((r[i] / np.sin(2 * r[i] * self.gamma)) ** 2) * (
                plus_variance + minus_variance
            )

        return variance

    def get_metadata(self):
        meta_dict = super().get_metadata()
        meta_dict["Gradient"] = "general Parameter-Shift"
        meta_dict["gamma"] = str(self.gamma)
        return meta_dict

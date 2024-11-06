# https://github.com/albaaparicio12/TFG-App/blob/38a589b4d9a96d0d71fe61cdd89093d2ce827021/src/business/extended/quantum_models/QSVCModel.py
from qiskit.circuit import ParameterVector
from custom_inherit import doc_inherit
import numpy as np
from src.business.base.QuantumModel import QuantumModel

# Package to evaluate model performance
from sklearn import metrics

from qiskit import QuantumCircuit

# Import a quantum feature map 
from qiskit.circuit.library import ZZFeatureMap

# Import the quantum kernel and the trainer
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer

# Import the optimizer for training the quantum kernel
from qiskit.algorithms.optimizers import SPSA

from qiskit_machine_learning.algorithms import QSVC


class QSVCModel(QuantumModel):

    def __init__(self, dataset, quantum_instance, backend) -> None:
        super(QSVCModel, self).__init__(dataset, quantum_instance, backend)

    @doc_inherit(QuantumModel.run, style="google")
    def run(self):
        output = {}
        # Get dataset
        X_train, y_train, X_test, y_test = self.dataset.get_data()

        # n_qubits will be the number of classes to clasify in the selected dataset
        n_qubits = 2

        # Define the Quantum Feature Map
        # Create a rotational layer to train. We will rotate each qubit the same amount.
        user_params = ParameterVector("θ", 1)
        fm0 = QuantumCircuit(n_qubits)
        for qubit in range(n_qubits):
            fm0.ry(user_params[0], qubit)

        # Use ZZFeatureMap to represent input data
        fm1 = ZZFeatureMap(n_qubits, reps=2)

        # Create the feature map, composed of our two circuits
        fm = fm0.compose(fm1)

        output['parameters'] = f"Trainable parameters: {user_params}"
        print(f"Trainable parameters: {user_params}")

        # Instantiate quantum kernel
        quantum_kernel = QuantumKernel(fm, user_parameters=user_params, quantum_instance=self._backend)
        fm.decompose().draw(output="mpl", filename='./static/files/circuit.png')
        """
        Since no analytical gradient is defined for kernel loss functions, gradient-based optimizers are not recommended for training kernels.
        """

        optimizer = SPSA(maxiter=30, learning_rate=0.05, perturbation=0.05)

        # Instantiate a quantum kernel trainer.
        qkt = QuantumKernelTrainer(
            quantum_kernel=quantum_kernel, loss="svc_loss", optimizer=optimizer, initial_point=[np.pi / 2],
        )

        # Train the kernel using QKT directly
        qka_results = qkt.fit(X_train, y_train)
        optimized_kernel = qka_results.quantum_kernel
        # Use QSVC for classification
        qsvc = QSVC(quantum_kernel=optimized_kernel)

        # Fit the QSVC
        qsvc.fit(X_train, y_train)

        # Predict the labels
        labels_test = qsvc.predict(X_test)

        # Evalaute the test accuracy
        accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=labels_test)
        print(f"accuracy test: {accuracy_test}")
        output['accuracy'] = f"Porcentaje de exactitud: {accuracy_test * 100}%"
        output['y_test'] = f"Valores reales: {y_test}"
        output['labels_test'] = f"Valores predecidos: {labels_test}"
        print(y_test)
        print(labels_test)
        imagenes = ['./static/files/data.png', './static/files/circuit.png']
        return output, imagenes

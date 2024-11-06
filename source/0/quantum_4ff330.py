# https://github.com/IsaacNez/qml/blob/8c1176681b4e2b8754f33108c7c73095c79d64f7/network/quantum.py
""" Class definition for the Quantum Operator """
import os
import numpy as np
from typing import *
import tensorflow as tf
import qiskit
from qiskit import Aer
from qiskit_aer import AerError
from qiskit.quantum_info.operators.predicates import (is_hermitian_matrix,is_unitary_matrix)
import gc

# tf.config.set_visible_devices([], 'GPU')

class QuantumOperator():
  def __init__(self, 
              circuit_dimension: int = 16,
              unitary_dimension: int = 4,
              debug_log: bool = False,
              draw_circuit: bool = False, 
              show_gpu_support: bool = False,
              enable_gpu: bool = True) -> 'QuantumOperator':
    """
    Defines the Quantum Operator.

    The Quantum Operator handles the transformation from the classical
    input into the quantum circuit.

    Args:
      circuit_dimension:  It defines how many qubits the circuit should
                          use. The default is 16 (= 4^2) because it assumes
                          an input image of 4x4.

      unitary_dimension:  It defines the input qubits to the unitary
                          operator. The default is 4 ( = 2^2) because it
                          assumes two-qubit unitary transformations.

                          The weight vector dimensions are calculated with
                          this parameter. The dimensions are defined by: 
                          circuit_dimension - 1 x unitary_dimension^2

      debug_log:          Enables debug logs for this class

      draw_circuit:       Enables drawing the circuit after it is built.
                          This function is not thread safe.

      show_gpu_support:   It enables debug logs to indicate if the backend
                          simulator supports Hardware Acceleration.

      enable_gpu:         It checks if the choosen backend supports GPU.
                          If GPU is enabled but the backend does not support 
                          it, it will run on the CPU. 
    """

    self.circuit_dimension = circuit_dimension
    self.unitary_dimension = unitary_dimension
    self.debug_log = debug_log
    self.draw_circuit = draw_circuit
    self.show_gpu_support = show_gpu_support
    self.enable_gpu = enable_gpu
  
  def feature_map(self, image: np.ndarray) -> np.ndarray:
    """ Maps a normalized input image to the feature map
        given by x -> |Φ> = [cos(x_1*pi/2) sin(x_1*pi/2) ] ⊗ ... ⊗  [cos(x_n*pi/2) sin(x_n*pi/2) ]
        where n is the number of pixels.

    Args:
      image:        Normalized and flatten image of size (N*N, ) 
  
    Output:
      ft_map:       Feature map of size (N*N, 2)
    """

    ft_map = np.zeros((image.shape[0], 2))
    for index, value in enumerate(image):
      ft_map[index][0] = np.cos(np.pi / 2 * value)
      ft_map[index][1] = np.sin(np.pi / 2 * value)

    return ft_map
  
  def hermitian_matrix(self, weights: tf.Tensor, unitary_dimension: int = 4) -> tf.Tensor:
    """ Generate a hermitian matrix using the weighs. 

        It generates a hermitian matrix from the weights where the diagonal are the first
        uniary_dimension elements. The strictly upper triangular indices are built by composing
        complex numbers from the remaining weights.

    Args:
      weights:              A tensor of size (unitary_dimension^2,) or ((unitary_dimension^2)^2,) 
                            for the normal/experimental and efficient circuit, respectively.          
      
      unitary_dimension:    Size of the diagonal of the resulting hermitian matrix. For a two-qubit unitary
                            gate, the matrix is (4, 4) = 4 diagonal elements. For a four-qubit unitary gate, 
                            the matrix is (16, 16) = 16 diagonal elements.

    Output:
      Tensor:               Resulting hermitian matrix of size (unitary_dimension, unitary_dimension)

    Raise:
      is_hermitian_matrix:  If the resulting matrix from the weights is not hermitian, it will raise an assert error.
    """
    diag = weights[:unitary_dimension]
    complex_range = (weights.shape[0] - unitary_dimension) // 2 + unitary_dimension
    reals = weights[unitary_dimension:complex_range]
    img = weights[complex_range:]

    assert reals.shape == img.shape

    diag_matrix = np.matrix(np.diag(diag.numpy()))

    hermitian = np.matrix(np.zeros((unitary_dimension, unitary_dimension), dtype=complex))
    hermitian[np.triu_indices(unitary_dimension, 1)] = np.array([complex(a, b) for a, b in zip(reals, img)])

    hermitian = hermitian + hermitian.H + diag_matrix

    assert is_hermitian_matrix(hermitian)
    return tf.convert_to_tensor(hermitian, dtype=tf.complex128)

  def unitary_matrices(self, weights: tf.Tensor = None, unitary_dimension: int = 4) -> tf.Tensor:
    """ Generate unitary matrices from hermitian matrices 

        From the array of weights, generate a per-weight hermitian matrix to then
        generate a unitary matrix by exp(1j*H).

    Args:
      weights:              A tensor of size A tensor of size (circuit_dimension -1, unitary_dimension^2) or 
                            (circuit_dimension -3, (unitary_dimension^2)^2) for the normal/experimental
                            and efficient circuit, respectively.  
      
      unitary_dimension:    The size of the resulting unitary matrix.  For a two-qubit unitary
                            gate, the matrix is (4, 4) = 4 diagonal elements. For a four-qubit unitary gate, 
                            the matrix is (16, 16) = 16 diagonal elements.
    
    Output:
      Tensor:               An unitary tensor of size (circuit_dimension -1, unitary_dimension, unitary_dimension) or 
                            (circuit_dimension -3, unitary_dimension^2, unitary_dimension^2) for the normal/experimental
                            and efficient circuit, respectively.
    
    Raise:
      is_unitary_matrix:    Raises an assert error if the resulting matrices are not unitary.
    """
    if weights == None:
      weights = self.weights

    unitaries = []
    
    for weight in weights:
      unitary = tf.linalg.expm(1j*self.hermitian_matrix(weight, unitary_dimension))
      assert is_unitary_matrix(unitary)
      unitaries.append(unitary)
      del unitary
    
    U = tf.convert_to_tensor(unitaries, dtype=tf.complex128)
    
    del unitaries
    gc.collect()
    return U
  

  def execute(self, image: tf.Tensor = None, 
                    backend: str = "aer_simulator", 
                    draw: bool = False,
                    output_format: str = "mpl",
                    filename: str = "qiskit_circuit", 
                    shots: int = 512,
                    weights: tf.Tensor = None,
                    circuit_type: str = 'normal',
                    device: str = "/physical_device:CPU:0") -> (dict):

    """ Creates the Quantum Circuit for binary classification based on 
        the given image and weights.

    Args:
      image:            The image to classify. The size must be (circuit_size,)
      
      backend:          Backend simulator to execute the quantum circuit.

      draw:             Indicate if the circuit should be drawn before 
                        executing the experiment. There is no option to 
                        draw this circuit interactively but rather it is
                        written to a file.

      output_format:    Format to draw the circuit. For me, look at the backend plotters
                        supported by Qiskit.

      filename:         Filename of the resulting drawing.

      shots:            The number of times to repeat the experiment.

      weights:          Model space parameters to influence the binary classification.

      circuit_type:     Type of circuit to build. Currently, we support three types of 
                        circuits:

                          1.  Normal (normal): this circuit uses (circuit_size - 1) two-qubit unitary gates
                              spread across log2(circuit_size) layers. This circuit expects
                              (circuit_size - 1)*(unitary_size^2) parameters. This circuit uses 
                              circuit_size qubits.

                          2.  Efficient (efficient): this circuit uses (circuit_size - 3) four-qubit unitary gates
                              spread across (circuit_size - 3) layers. This circuit uses only 4 qubits
                              and one classical register but it uses (circuit_size - 1)*(unitary_size^2)^2
                              parameters

                          3.  Experimental (experimental): this circuit is my own implementation. It uses 
                              (circuit_size -1) two-qubit unitary gates spread over
                              (circuit_size - 4)/2 layers but it uses only 4 qubits and one classical
                              register. This circuit uses (circuit_size - 1)*(unitary_size^2) trainable parameters
      
      device:           It specifies the device to place the data. For GPU-enable systems, it offers Hardware 
                        Acceleration for Tensorflow operations. By default, it uses the CPU.

    Output:
      counts:           It returns the result from the binary classification with the counts per class in the form
                        {"0": x, "1": y}, where x + y = shots. If there is an error, it will return None.

    Raises:
      ValueError:       - If image is not given, it will raise a ValueError since it is needed to calculate the feature map.
                        - If the filename is not given (when drawing the circuit), it will raise the exception since the class
                          does not support saving to empty filenames.
                        - If the circuit_type is not one of the three allowed names, it will raise the exception since it is 
                          unknown behavior.
    """
    if image is None:
      raise ValueError("The image cannot be None. Please pass an image.")
    
    if len(filename) <= 0 and draw:
      raise ValueError("Please indicate a name for the filename")

    tf.device(device)

    if weights is None:
      if circuit_type == 'normal' or circuit_type == 'experimental':
        self.weights = tf.random.normal((self.circuit_dimension - 1, self.unitary_dimension ** 2))
      else:
        self.weights = tf.random.normal((self.circuit_dimension - 3, (self.unitary_dimension * 2) ** 2))
    else:
      self.weights = weights

    feature_map = self.feature_map(image.numpy().flatten())

    if circuit_type == 'normal' or circuit_type == 'experimental':
      unitaries = self.unitary_matrices().numpy()
    else:
      unitaries = self.unitary_matrices(unitary_dimension=self.unitary_dimension**2).numpy()

    if circuit_type == 'efficient':
      quantum_circuit = self.gen_efficient_circuit(feature_map, unitaries)
    elif circuit_type == 'normal':
      quantum_circuit = self.gen_normal_circuit(feature_map, unitaries)
    elif circuit_type == 'experimental':
      quantum_circuit = self.gen_experimental_circuit(feature_map, unitaries)
    else:
      raise ValueError(f"circuit_type can only be normal, efficient or experimental. We received: {circuit_type}")

    if self.draw_circuit or draw:
      fig = quantum_circuit.draw(output=output_format, filename=filename)
      fig.clf()

    try:
      if backend == "aer_simulator":
        circuit_backend = Aer.get_backend(backend)
        if 'GPU' in circuit_backend.available_devices() and self.enable_gpu:
          circuit_backend = Aer.get_backend(backend, device='GPU')
          if self.show_gpu_support:
            print(f"The backend {backend} supports GPU. We are using it!")
        else:
          circuit_backend = Aer.get_backend(backend)
          if self.show_gpu_support:
            print(f"The backend {backend} supports GPU. We are ignoring it...")
      else:
        circuit_backend = Aer.get_backend(backend)
        if self.show_gpu_support:
            print(f"Your backend {backend} does not support GPU. We are ignoring it...")
      
      counts = qiskit.execute(quantum_circuit, circuit_backend, shots=shots).result().get_counts()

      del quantum_circuit
      del feature_map
      del unitaries
      del circuit_backend
      gc.collect()

      return counts
    except AerError as e:
      print(f"This module generated the following error [{e}]")
      return None


  def gen_normal_circuit(self, features: tf.Tensor, unitaries: tf.Tensor) -> qiskit.QuantumCircuit:
    """ Generates the 'normal' Quantum Circuit 

    Args:
      features:         Tensor representing the feature map for the image to classify.

      unitaries:        Tensor containing the description of the unitary gates for the circuit.

    Output:
      QuantumCircuit:   Quantum Circuit of size with circuit_dimension qubits and circuit_dimension -1 
                        unitary gates over log2(circuit_dimension) layers.
    """
    quantum_circuit = qiskit.QuantumCircuit(self.circuit_dimension, 1)

    for index, feature in enumerate(features):
      quantum_circuit.initialize(feature, index)
    
    index = 0
    for layer in range(int(np.log2(self.circuit_dimension))):
      for lower in range((2**layer -1 if layer > 1 else layer), self.circuit_dimension, (2**(layer+1))):
        upper = lower + layer + 1 if layer < 2 else lower + 2**layer

        quantum_circuit.unitary(unitaries[index], [quantum_circuit.qubits[lower], quantum_circuit.qubits[upper]], f"$U_{{{index}}}$")
        index += 1
    
    quantum_circuit.measure([self.circuit_dimension - 1], [0])

    return quantum_circuit

  def gen_efficient_circuit(self, features, unitaries):
    """ Generates the efficient Quantum Circuit 

    Args:
      features:         Tensor representing the feature map for the image to classify.

      unitaries:        Tensor containing the description of the unitary gates for the circuit.

    Output:
      QuantumCircuit:   Quantum Circuit of size with 4 qubits and circuit_dimension - 3 
                        unitary gates over circuit_dimension - 3 layers.
    """
    quantum_circuit = qiskit.QuantumCircuit(4, 1)
    quantum_circuit.initialize(features[0], 0)
    quantum_circuit.initialize(features[1], 1)
    quantum_circuit.initialize(features[2], 2)
    quantum_circuit.initialize(features[3], 3)
    
    quantum_circuit.unitary(unitaries[0], quantum_circuit.qubits[0:4], f'$U_{0}$')

    for index, feat in enumerate(features[4:]):
      quantum_circuit.reset(3)
      quantum_circuit.initialize(feat, 3)
      quantum_circuit.unitary(unitaries[index + 1], quantum_circuit.qubits[0:4], f"$U_{{{index + 1}}}$")
    
    quantum_circuit.measure([0], [0])

    return quantum_circuit
  
  def gen_experimental_circuit(self, features, unitaries):
    """ Generates the experimental Quantum Circuit 

    Args:
      features:         Tensor representing the feature map for the image to classify.

      unitaries:        Tensor containing the description of the unitary gates for the circuit.

    Output:
      QuantumCircuit:   Quantum Circuit of size with 4 qubits and circuit_dimension - 1 
                        unitary gates over (circuit_dimension - 4)/2 layers.
    """
    quantum_circuit = qiskit.QuantumCircuit(4,1)
    quantum_circuit.initialize(features[0], 0)
    quantum_circuit.initialize(features[1], 1)
    quantum_circuit.initialize(features[2], 2)
    quantum_circuit.initialize(features[3], 3)
    
    quantum_circuit.unitary(unitaries[0], quantum_circuit.qubits[0:2], f'$U_{0}$')
    quantum_circuit.unitary(unitaries[1], quantum_circuit.qubits[2:4], f'$U_{1}$')

    layers = (self.circuit_dimension - 4) // 2

    reset_indexes = [[0,2],[1,3]]

    idx = 0
    for index in range(0, layers):
      quantum_circuit.reset(reset_indexes[idx%2][0])
      quantum_circuit.reset(reset_indexes[idx%2][1])
      quantum_circuit.initialize(features[2*index + 4], reset_indexes[idx%2][0])
      quantum_circuit.initialize(features[2*index + 5], reset_indexes[idx%2][1])
      quantum_circuit.unitary(unitaries[2*index + 2], quantum_circuit.qubits[0:2], f'$U_{{{2*index + 2}}}$')
      quantum_circuit.unitary(unitaries[2*index + 3], quantum_circuit.qubits[2:4], f'$U_{{{2*index + 3}}}$')
      # idx += 1
    
    quantum_circuit.unitary(unitaries[-1], [quantum_circuit.qubits[1],quantum_circuit.qubits[3]], f'$U_{{{2*index + 4}}}$')

    quantum_circuit.measure([3], [0])

    return quantum_circuit

  def plot_circuits(self, image_size: int = 4, circuits: list = ["normal"], path: str = "circuits/", filename: str = "circuit", extension: str = "png", output_format: str = "mpl") -> None:
    """ Only plot function for the Quantum Operator

        Plot a generic version of the circuits. It creates a Gaussian distributed image 
        as the input. 

    Args:
      image_size:       Size of the image to generate.

      circuits:         Type of circuits to generate. It accepts 'normal', 'efficient',
                        and 'experimental'.

      path:             Path where the circuit images must be stored. It accepts both relative
                        and absolute paths. Before saving the drawings, it will check if the path
                        exist, if it does not, it will be created.

      filename:         Base filename of the resulting circuit. The final filename will be composed by
                        this argument, the circuit type, the image size, and the extension.

      extension:        It defines the type of file to generate. Be aware the output_format must support
                        your extension. For more info, refer to Qiskit documentation.

      output_format:    The backend to render the circuit. It is passed down to Qiskit.
    """
    self.circuit_dimension = image_size*image_size
    for circuit_type in circuits:
      print(f"Creating the {circuit_type} circuit")
      if circuit_type == 'normal' or circuit_type == 'experimental':
        weights = tf.random.normal((self.circuit_dimension - 1, self.unitary_dimension ** 2))
        unitaries = self.unitary_matrices(weights=weights).numpy()
      else:
        weights = tf.random.normal((self.circuit_dimension - 3, (self.unitary_dimension ** 2) ** 2))
        unitaries = self.unitary_matrices(weights=weights, unitary_dimension=self.unitary_dimension**2).numpy()

      image = np.random.normal(size=(image_size, image_size))
      ft_map = self.feature_map(image.flatten())

      if circuit_type == 'efficient':
        quantum_circuit = self.gen_efficient_circuit(ft_map, unitaries)
      elif circuit_type == 'normal':
        quantum_circuit = self.gen_normal_circuit(ft_map, unitaries)
      elif circuit_type == 'experimental':
        quantum_circuit = self.gen_experimental_circuit(ft_map, unitaries)
      else:
        raise ValueError(f"circuit_type can only be normal, efficient or experimental. We received: {circuit_type}")

      abs_path = os.path.abspath(path)
      if not os.path.exists(abs_path):
        os.makedirs(abs_path)

      abs_filename = os.path.join(abs_path, f"{filename}_{circuit_type}_{image_size}x{image_size}.{extension}")
      fig = quantum_circuit.draw(output=output_format, filename=abs_filename)
      fig.clf()






# https://github.com/lslattery11/quantum_kernel/blob/93505d3cec53b59b5d84697ef5542e270719963d/code/quantum_kernels/projected_kernels.py
import numpy as np
import itertools
from typing import Optional, Union, Sequence, Mapping, List, Set
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit.circuit import Parameter,ParameterVector
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
import qiskit.quantum_info as qi


class RDM1ProjectedKernel(QuantumKernel):
    """
    Projected quantum kernel using a single qubit RDM. See Eq. L6 in https://arxiv.org/abs/2011.01938.
    additional arguments for gamma (kernel bandwith) and measure_qubits. 
    """
    def __init__(
        self,
        feature_map: Optional[QuantumCircuit] = None,
        enforce_psd: bool = True,
        batch_size: int = 900,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        user_parameters: Optional[Union[ParameterVector, Sequence[Parameter]]] = None,
        gamma: float = 1.0,
        measured_qubits: List[List[int]] = [[0]],
    ) -> None:
        super().__init__(feature_map,enforce_psd,batch_size,quantum_instance,user_parameters)
        self.gamma=gamma
        self._measured_qubits=measured_qubits

    def set_measured_qubits(self,qubits:List[List[int]]):
        self._measured_qubits=qubits

    def construct_circuit(self,
        x: ParameterVector,
        measurement: bool = True,
        is_statevector_sim: bool = False,
    ):
        """
        Overrides parent class function. Parent function returns a single circuit for computing overlap.
        This is a modified version where I will return multiple quantum circuits used in computing the reduced
        density matrix of a single qubit.
        """
        # Use the parent class checks.
        # Ensure all user parameters have been bound in the feature map circuit.
        unbound_params = self.get_unbound_user_parameters()
        if unbound_params:
            raise ValueError(
                f"""
                The feature map circuit contains unbound user parameters ({unbound_params}).
                All user parameters must be bound to numerical values before constructing
                inner product circuit.
                """
            )

        if len(x) != self._feature_map.num_parameters:
            raise ValueError(
                "x and class feature map incompatible dimensions.\n"
                f"x has {len(x)} dimensions, but feature map has {self._feature_map.num_parameters}."
            )
        #create circuit for data x
        q = QuantumRegister(self._feature_map.num_qubits, "q")
        c = ClassicalRegister(self._feature_map.num_qubits, "c")
        qc = QuantumCircuit(q, c,name='qcx')
        x_dict = dict(zip(self._feature_map.parameters, x))
        psi = self._feature_map.assign_parameters(x_dict)
        qc.append(psi.to_instruction(), qc.qubits)
        #add measurements to the appropriate qubits.
        qcs=[]
        qc.barrier(q)
        
        for i,j in enumerate(self._measured_qubits):
            new_qc=qc.copy(name=qc.name+str(i))
            if not is_statevector_sim:  
                new_qc.measure(j,j)
            qcs.append(new_qc)
        
        return qcs


    def _compute_overlap(self,
        idx_x,
        idx_y,
        result_info,
        is_statevector_sim,
        measurement_basis0: Optional[str] = None,
        measurement_basis1: Optional[str] = None,
        ) -> float:
        """
        Overrides parent class function. Instead of computing overlap, we wil compute
        the projected kernel element. Currently not doing a RDM but instead <Zx>-<Zy> for the rdm.
        """
        if is_statevector_sim:
            sum_norms=0
            subystems=self._measured_qubits
            for i in range(len(subystems)):
                #result info for statevector simulation should be the rdms.
                rdmx=result_info[idx_x,i]
                rdmy=result_info[idx_y,i]
                sum_norms+=np.linalg.norm(rdmx-rdmy,ord='fro')**2
            #kernel_value=np.exp(-self.gamma*sum_norms)
            kernel_value=sum_norms
        else:
            count_x = result_info[idx_x]
            count_y = result_info[idx_y]

            Zs=[]
            for count in [count_x,count_y]:
                try:
                    up=count[measurement_basis0]
                except:
                    up=0
                try:
                    down=count[measurement_basis1]
                except:
                    down=0
                Z=(up-down) / (up+down)
                Zs.append(Z)
            #now compute kernel
            #kernel_value=np.exp(-self.gamma*np.linalg.norm(Zs[0]-Zs[1])**2)
            kernel_value=np.linalg.norm(Zs[0]-Zs[1])
        return kernel_value

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None,return_rdms=False):
        r"""
        Modified version of the parent class function in order to work for projected quantum kernels.
        Construct kernel matrix for given data and feature map

        If y_vec is None, self inner product is calculated.
        If using `statevector_simulator`, only build circuits for :math:`\Psi(x)|0\rangle`,
        then perform inner product classically.

        Args:
            x_vec: 1D or 2D array of datapoints, NxD, where N is the number of datapoints,
                                                            D is the feature dimension
            y_vec: 1D or 2D array of datapoints, MxD, where M is the number of datapoints,
                                                            D is the feature dimension
            return_rdm boolean indicating whether to return rdms or not.

        Returns:
            2D matrix, NxM

        Raises:
            QiskitMachineLearningError:
                - A quantum instance or backend has not been provided
            ValueError:
                - unbound user parameters in the feature map circuit
                - x_vec and/or y_vec are not one or two dimensional arrays
                - x_vec and y_vec have have incompatible dimensions
                - x_vec and/or y_vec have incompatible dimension with feature map and
                    and feature map can not be modified to match.
        """
        # Ensure all user parameters have been bound in the feature map circuit.
        unbound_params = self.get_unbound_user_parameters()
        if unbound_params:
            raise ValueError(
                f"""
                The feature map circuit contains unbound user parameters ({unbound_params}).
                All user parameters must be bound to numerical values before evaluating
                the kernel matrix.
                """
            )

        if self._quantum_instance is None:
            raise QiskitMachineLearningError(
                "A QuantumInstance or Backend must be supplied to evaluate a quantum kernel."
            )
        if isinstance(self._quantum_instance, Backend):
            self._quantum_instance = QuantumInstance(self._quantum_instance)

        if not isinstance(x_vec, np.ndarray):
            x_vec = np.asarray(x_vec)
        if y_vec is not None and not isinstance(y_vec, np.ndarray):
            y_vec = np.asarray(y_vec)

        if x_vec.ndim > 2:
            raise ValueError("x_vec must be a 1D or 2D array")

        if x_vec.ndim == 1:
            x_vec = np.reshape(x_vec, (-1, len(x_vec)))

        if y_vec is not None and y_vec.ndim > 2:
            raise ValueError("y_vec must be a 1D or 2D array")

        if y_vec is not None and y_vec.ndim == 1:
            y_vec = np.reshape(y_vec, (-1, len(y_vec)))

        if y_vec is not None and y_vec.shape[1] != x_vec.shape[1]:
            raise ValueError(
                "x_vec and y_vec have incompatible dimensions.\n"
                f"x_vec has {x_vec.shape[1]} dimensions, but y_vec has {y_vec.shape[1]}."
            )

        if x_vec.shape[1] != self._feature_map.num_parameters:
            try:
                self._feature_map.num_qubits = x_vec.shape[1]
            except AttributeError:
                raise ValueError(
                    "x_vec and class feature map have incompatible dimensions.\n"
                    f"x_vec has {x_vec.shape[1]} dimensions, "
                    f"but feature map has {self._feature_map.num_parameters}."
                ) from AttributeError

        if y_vec is not None and y_vec.shape[1] != self._feature_map.num_parameters:
            raise ValueError(
                "y_vec and class feature map have incompatible dimensions.\n"
                f"y_vec has {y_vec.shape[1]} dimensions, but feature map "
                f"has {self._feature_map.num_parameters}."
            )

        # determine if calculating self inner product
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        # initialize kernel matrix
        kernel = np.zeros((x_vec.shape[0], y_vec.shape[0]))

        # set diagonal to 1 if symmetric
        # here is a major change!!!!!!. Projected kernels measure distance not similarity so diagonal should be 0.
        if is_symmetric:
            np.fill_diagonal(kernel, 0)

        # get indices to calculate
        if is_symmetric:
            mus, nus = np.triu_indices(x_vec.shape[0], k=1)  # remove diagonal
        else:
            mus, nus = np.indices((x_vec.shape[0], y_vec.shape[0]))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        is_statevector_sim = self._quantum_instance.is_statevector
        measurement = not is_statevector_sim
        measurement_basis0 = "0"*self._feature_map.num_qubits
        #bit flip for all measured qubits so that I can take <Zi...Zj> on the measured qubits. Note reverse of range(num_qubits) is used b/c qiskit has reverse convention
        #than every paper, textbook & sane person ever.
        measurement_basis1=""
        measurement_basis1=measurement_basis1.join(["1" if idx in self._measured_qubits else "0" for idx in range(self._feature_map.num_qubits-1,-1,-1)])

        if is_symmetric:
            to_be_computed_data = x_vec
        else:  # not symmetric
            to_be_computed_data = np.concatenate((x_vec, y_vec))
        #here we actually just want to compute the rdm tr_a(|psi><psi|O) for each unique data vector. Original design
        #was for each pair to compute the reduced density matrices. but instead what we should do is just compute once and then pair up appropriately to form the kernel matrix.
        #for now we will let O = Zi x Zj x Zk (i.e. a tensor product of z operators on the ith, jth and kth qubits) in order to get a working model.
        #To change this we'd have to change which circuits we are constructing (i.e. add relevant rotation before a qubit in construct_circuit and change how the circuits are paired up when we want to compute the overlap)
        # calculate kernel
        if is_statevector_sim:  # using state vector simulator
            feature_map_params = ParameterVector("par", self._feature_map.num_parameters)
            #currently parameterized_circuits is just = [circuit] but will likely want to change in future. 
            parameterized_circuits = self.construct_circuit(
                feature_map_params,
                measurement=measurement,
                is_statevector_sim=is_statevector_sim,
            )

            parameterized_circuits = self._quantum_instance.transpile(
                parameterized_circuits, pass_manager=self._quantum_instance.unbound_pass_manager
            )

            #For the projected kernel we only need to compute the circuit(s) expectation value for each data point
            #and then construct the kernel matrix from K(<Ox>,<Oy>)
            
            #qubit subsystems; we will take the trace over complement.
            subsystems=self._measured_qubits
            system=set(range(self._feature_map.num_qubits))
            complements=[[element for element in system if element not in subsystem] for subsystem in subsystems]
            rdms=np.zeros((len(to_be_computed_data),len(subsystems)),dtype=object)

            for min_idx in range(0, len(to_be_computed_data), self._batch_size):
                max_idx = min(min_idx + self._batch_size, len(to_be_computed_data))
                circuits = [
                        parameterized_circuits[idx].assign_parameters(
                            {feature_map_params: x}
                        )
                        for idx in range(len(parameterized_circuits))

                        for x in to_be_computed_data[min_idx:max_idx]
                    ]

                if self._quantum_instance.bound_pass_manager is not None:
                        circuits = self._quantum_instance.transpile(
                            circuits, pass_manager=self._quantum_instance.bound_pass_manager
                        )
                        
                results = self._quantum_instance.execute(circuits, had_transpiled=True)
                #we should store rdms over for loop not statevectors due to large memory size of statevectors.
                for i,j in itertools.product(range(min_idx,max_idx),range(len(complements)),repeat=1):
                    rdms[i,j]=np.array(qi.partial_trace(results.get_statevector(i-min_idx),complements[j]))

            matrix_elements = [
                    self._compute_overlap(i, j, rdms, is_statevector_sim)
                    for i,j in zip(mus,nus)
                ]
        else:  # not using state vector simulator
            raise NotImplementedError('Only statevector support is currently implemented.')
            feature_map_params = ParameterVector("par", self._feature_map.num_parameters)
            #currently parameterized_circuits is just = [circuit] but will likely want to change in future. 
            parameterized_circuits = self.construct_circuit(
                feature_map_params,
                measurement=measurement,
                is_statevector_sim=is_statevector_sim,
            )

            parameterized_circuits = self._quantum_instance.transpile(
                parameterized_circuits, pass_manager=self._quantum_instance.unbound_pass_manager
            )
            #For the projected kernel we only need to compute the circuit(s) expectation value for each data point
            #and then construct the kernel matrix from K(<Ox>,<Oy>)
            counts=[]
            for min_idx in range(0, len(to_be_computed_data), self._batch_size):

                circuits = [
                        parameterized_circuits[idx].assign_parameters(
                            {feature_map_params: x}
                        )
                        for idx in range(len(parameterized_circuits))

                        for x in to_be_computed_data
                    ]

                if self._quantum_instance.bound_pass_manager is not None:
                        circuits = self._quantum_instance.transpile(
                            circuits, pass_manager=self._quantum_instance.bound_pass_manager
                        )
                        
                results = self._quantum_instance.execute(circuits, had_transpiled=True)
                counts=counts+results.get_counts()

            matrix_elements = [
                self._compute_overlap( i, j, counts, is_statevector_sim, measurement_basis0, measurement_basis1)

                for i,j in zip(mus,nus)

                ]

        for i, j, value in zip(mus, nus, matrix_elements):
            kernel[i, j] = value
            if is_symmetric:
                kernel[j, i] = kernel[i, j]

        if self._enforce_psd and is_symmetric and not is_statevector_sim:
            # Find the closest positive semi-definite approximation to symmetric kernel matrix.
            # The (symmetric) matrix should always be positive semi-definite by construction,
            # but this can be violated in case of noise, such as sampling noise, thus the
            # adjustment is only done if NOT using the statevector simulation.
            D, U = np.linalg.eig(kernel) 
            kernel = U @ np.diag(np.maximum(0, D)) @ U.transpose()

        if return_rdms==True:
            return np.real(kernel),rdms
        else:
            return np.real(kernel)

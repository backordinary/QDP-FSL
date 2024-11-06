# https://github.com/yiiyama/qc_random/blob/47e125d64d93d0c4472cf4ac5cb6f8b2e6eb8ec7/quantum_mlp/quantum_mlp.py
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library.basis_change.qft import QFT
import numpy as np

 # Flag to not leave e.g. ancillas in used state
CLEAN_UP_AFTER_ONESELF = True

##########################
### The main function ###
##########################

def binary_mlp(layer_nodes, input_x, input_y, x_precision=1, gatify=-1):
    """Sets up the circuit implementing a binary-weight MLP classifier

    Args:
        layer_nodes (list[int]): Number of neurons in layers 0-(N-1).
        input_x (np.ndarray(shape=(num_samples, layer_nodes[0]), dtype=int)): Values 0 to 2^(x_precision) - 1.
        input_y (np.ndarray(shape=(num_samples,), dtype=int)): Values 0 or 1
        x_precision (int): Precision of input x.
        gatify (int): Abstraction level for component functions.
        
    Returns:
        QuantumCircuit: A circuit implementing the MLP maximizing sum_{sample}[output * (y - 0.5)]
    """
    
    assert(input_x.shape[0] == input_y.shape[0] and input_x.shape[1] == layer_nodes[0])
    assert(x_precision > 0)
    
    reg_input_index, reg_data_input_y, reg_data, reg_weights, reg_biases, reg_amp = register_lists = setup_registers(layer_nodes, input_x, input_y, x_precision)

    registers = flatten_list(*register_lists)
    circuit = QuantumCircuit(*registers)

    circuit += initialize_weights(reg_weights, reg_biases, gatify=gatify)
    circuit += load_data(reg_input_index, reg_data[0], reg_data_input_y[0], input_x, input_y, x_precision, gatify=gatify)
    if gatify >= 0:
        circuit.barrier()
        
    circuit += feedforward(reg_data, reg_weights, reg_biases, gatify=gatify)
    if gatify >= 0:
        circuit.barrier()

    circuit += transduce_amplitude(reg_data_input_y[0], reg_data[-1][0], reg_amp[0], gatify=gatify)
    if gatify >= 0:
        circuit.barrier()
        
    return circuit


###########################
### Component functions ###
###########################

def to_gate(c, label):
    """Make a circuit where the argument is made into a composite gate.
    
    Args:
        c (QuantumCircuit): Circuit to turn into a gate.
        label (str): Gate name.

    Returns:
        QuantumCircuit: The circuit with the same set of registers as the input and a single gate
            corresponding to the input.
    """
    
    circuit = QuantumCircuit(*c.qregs)
    qubits = sum((reg[:] for reg in c.qregs), [])
    circuit.append(c.to_gate(label=label), qubits)
    return circuit


def setup_registers(layer_nodes, input_x, input_y, x_precision):
    log2_num_samples = np.log2(input_x.shape[0])
    if log2_num_samples != np.floor(log2_num_samples):
        raise RuntimeError('Can only handle input array with sizes in powers of 2')
        
    input_index_size = log2_num_samples.astype(int)
    
    num_layers = len(layer_nodes)
    full_layer_nodes = list(layer_nodes)
    full_layer_nodes.append(1)
    
    reg_input_index = QuantumRegister(input_index_size, 'index')
    reg_data_input_x = list(QuantumRegister(x_precision, 'x^{{{}}}'.format(i)) for i in range(layer_nodes[0]))
    reg_data_input_y = QuantumRegister(1, 'y')

    reg_data = [reg_data_input_x]
    reg_weights = []
    reg_biases = []
    data_width = x_precision + 1
    data_max = 2 ** (x_precision - 1)
    for ilayer in range(num_layers):
        # fixing the bias width to the data width of the previous layer - can be reconsidered
        reg_biases.append(list(QuantumRegister(data_width, 'b^{{l{}n{}}}'.format(ilayer + 1, n)) for n in range(full_layer_nodes[ilayer + 1])))
        
        # could also think about truncating the layer inputs if compounding data size becomes an issue
        data_max *= (layer_nodes[ilayer] + 1) # +1 for the bias
        data_width = int(np.ceil(np.log2(data_max + 0.5))) + 1 # 1 for sign

        reg_data.append(list(QuantumRegister(data_width, 'd^{{l{}n{}}}'.format(ilayer + 1, n)) for n in range(full_layer_nodes[ilayer + 1])))

        weights = []
        for inode_next in range(full_layer_nodes[ilayer + 1]):
            weights.append(list(QuantumRegister(1, 'w^{{l{}n{}fromn{}}}'.format(ilayer + 1, inode_next, n)) for n in range(full_layer_nodes[ilayer])))
        reg_weights.append(weights)

    reg_amp = QuantumRegister(1, 'amp')
    
    return reg_input_index, reg_data_input_y, reg_data, reg_weights, reg_biases, reg_amp


def flatten_list(*reg_lists):
    registers = []
    for regs in reg_lists:
        if type(regs) is list:
            while type(regs[0]) is list:
                regs = sum(regs, [])
                
            registers += regs
        else:
            registers.append(regs)

    return registers


def initialize_weights(reg_weights, reg_biases, gatify=-1):
    """Initialize the weight registers into a full superposition state.
    
    Args:
        reg_weights (list[list[list[QuantumRegister]]]): weight registers
        reg_biases (list[list[QuantumRegister]]): bias registers
        
    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """
    
    registers = flatten_list(reg_weights, reg_biases)
    circuit = QuantumCircuit(*registers)
    
    # Make weights a full superposition
    for reg in registers:
        circuit.h(reg)

    if gatify != 0:
        return to_gate(circuit, 'initialize_weights')
    else:
        return circuit


def to_binary(arr, width):
    """Convert an array of integers with shape S into an array of uint8 with shape (S + [n])
    where n is the number of uint8s needed to accommodate `width`-wide integers
    
    Args:
        arr (np.ndarray): Input array.
        width (int): Width of the input (specification of input maximum as 2**width).
        
    Returns:
        np.ndarray: Binarized array.
    """

    ndiv = (width // 8) + 1
    if width % 8 == 0:
        ndiv -= 1
        
    if isinstance(arr, int):
        arr = np.array(arr)
        
    uint8_arr = np.empty(arr.shape + (ndiv,), dtype=np.uint8)
    for idiv in range(ndiv):
        uint8_arr[..., idiv] = (arr >> (8 * idiv)) & 0xff
        
    binary_arr = np.unpackbits(uint8_arr, axis=-1, bitorder='little')[..., :width]
    
    return binary_arr


def load_sample_x(reg_data_input_x, binary_x, gatify=-1):
    """Load the input x data of a sample into the register.
    
    Args:
        reg_data_input_x (QuantumRegister): Register to store the x values.
        x (np.ndarray): Value of x for a sample in binary (shape [ndim, precision])
        
    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """
    
    reg_ancilla = AncillaRegister(1, 'ancilla')
    ancilla = reg_ancilla[0]

    registers = flatten_list(reg_data_input_x, reg_ancilla)
    circuit = QuantumCircuit(*registers)

    for idim in range(binary_x.shape[0]):
        reg = reg_data_input_x[idim]
        for bit in binary_x[idim].nonzero()[0]:
            circuit.cx(ancilla, reg[bit])
            
    if gatify < 0 or gatify >= 1:
        return to_gate(circuit, 'load_sample_x')
    else:
        return circuit


def load_sample(reg_input_index, reg_data_input_x, data_input_y, binary_index, binary_x, y, gatify=-1):
    """Load the x and y data of a sample into the registers.
    
    Args:
        reg_input_index (QuantumRegister): Index register.
        reg_data_input_x (QuantumRegister): Register to store the x values.
        data_input_y (Qubit): Qubit to store the y values.
        binary_index (np.ndarray): Binarized index of the sample.
        binary_x (np.ndarray): Binarized x value of the sample.
        y (int): Y value.
        
    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """
    
    reg_ancilla = AncillaRegister(1, 'ancilla')
    ancilla = reg_ancilla[0]

    registers = flatten_list(reg_input_index, reg_data_input_x, data_input_y.register, reg_ancilla)
    circuit = QuantumCircuit(*registers)
    
    for bit in (1 - binary_index).nonzero()[0]:
        circuit.x(reg_input_index[bit])

    circuit.mcx(reg_input_index, ancilla)

    circuit += load_sample_x(reg_data_input_x, binary_x, gatify=gatify)

    if y == 1:
        circuit.cx(ancilla, data_input_y)

    circuit.mcx(reg_input_index, ancilla)

    for bit in (1 - binary_index).nonzero()[0]:
        circuit.x(reg_input_index[bit])
        
    if gatify < 0 or gatify >= 2:
        return to_gate(circuit, 'load_sample')
    else:
        return circuit

    
def load_data(reg_input_index, reg_data_input_x, data_input_y, input_x, input_y, x_precision=1, gatify=-1):
    """Load the x and y data into registers.
    
    Args:
        reg_input_index (QuantumRegister): register indexing the input samples.
        reg_data_input_x (list[QuantumRegister]): register storing the input x values.
        data_input_y (Qubit): Qubit storing the input y values.
        input_x (np.ndarray): input x values.
        input_y (np.ndarray): input y values.
        x_precision (int): precision of input x.
        
    Returns:
        QuantumCircuit: circuit corresponding to this subroutine.
    """
    
    registers = flatten_list(reg_input_index, reg_data_input_x, data_input_y.register)
    circuit = QuantumCircuit(*registers)

    circuit.h(reg_input_index)
    
    index_width = np.log2(input_x.shape[0]).astype(int)
    indices = np.arange(input_x.shape[0], dtype=int)
    
    binary_indices = to_binary(indices, index_width)

    binary_x = to_binary(input_x, x_precision)

    for idx in range(input_x.shape[0]):
        binary_index = binary_indices[idx]
        x = binary_x[idx]
        y = input_y[idx]
        
        circuit += load_sample(reg_input_index, reg_data_input_x, data_input_y, binary_index, x, y, gatify=gatify)
            
    if gatify < 0 or gatify >= 3:
        return to_gate(circuit, 'load_data')
    else:
        return circuit


def add_weighted_input_single(source_bit, weight, targ, sign_bit, gatify=-1):
    """Multiply one bit from the output of a node in the previous layer with the weight and
    pass the result to the target node.
    
    Args:
        source_bit (Qubit): Source bit.
        weight (Qubit): Binary weight connecting the input and target nodes.
        targ (QuantumRegister): Input source for the target node.
        sign_bit (Qubit or None): If a qubit, control the entire operation on this bit
            
    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """
    reg_ancilla = AncillaRegister(1, 'ancilla')
    ancilla = reg_ancilla[0]
    reg_ancilla_activation = AncillaRegister(1, 'activation')
    ancilla_activation = reg_ancilla_activation[0]
    circuit = QuantumCircuit(source_bit.register, weight.register, targ, reg_ancilla, reg_ancilla_activation)
    
    if sign_bit is not None:
        # because of x(sign_bit) in the parent function, sign_bit is 1 if sign of the source is positive
        circuit.mcx([source_bit, sign_bit], ancilla_activation)
        source_ctrl = ancilla_activation
    else:
        source_ctrl = source_bit

    dphi = 2. * np.pi / (2 ** targ.size)
        
    for itarg in range(targ.size):
        circuit.mcx([source_ctrl, targ[itarg]], ancilla)
        circuit.crz(2. * dphi * (2 ** (source_bit.index + itarg)), ancilla, weight) # note rz(2*theta) = exp(-theta Z)
        circuit.mcx([source_ctrl, targ[itarg]], ancilla)

    if sign_bit is not None:
        circuit.mcx([source_bit, sign_bit], ancilla_activation)
        
    if gatify < 0 or gatify >= 1:
        return to_gate(circuit, 'add_weighted_input_single')
    else:
        return circuit

    
def add_weighted_input(source, weight, targ, activation, gatify=-1):
    """Multiply the output of a node in the previous layer with the weight and
    pass the result to the target node.
    
    Args:
        source (QuantumRegister): Value of the source node.
        weight (Qubit): Binary weight connecting the input and target nodes.
        targ (QuantumRegister): Value of the target node.
        activation (bool): If True, last bit of source is considered as the sign bit
            and source is propagated only if this bit is 0.
            
    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """

    circuit = QuantumCircuit(source, weight.register, targ)
    
    if activation:
        max_digits = source.size - 1
        sign_bit = source[-1]
        circuit.x(sign_bit) # relu activation -> only apply rz when the sign bit is 0
    else:
        max_digits = source.size
        sign_bit = None
        
    for idigit in range(max_digits):
        circuit += add_weighted_input_single(source[idigit], weight, targ, sign_bit, gatify=gatify)
        
    if activation:
        circuit.x(sign_bit) # relu activation -> only apply rz when the sign bit is 0
        
    if gatify < 0 or gatify >= 2:
        return to_gate(circuit, 'add_weighted_input')
    else:
        return circuit

    
def add_bias(bias, targ, gatify=-1):
    """Apply the bias to the target node.
    
    Args:
        bias (QuantumRegister): Bias on the target node.
        targ (QuantumRegister): Value of the target node.

    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.    
    """

    reg_ancilla = AncillaRegister(1, 'ancilla')
    ancilla = reg_ancilla[0]
    circuit = QuantumCircuit(bias, targ, reg_ancilla)
    
    sign_bit = bias[-1]
    
    dphi = 2. * np.pi / (2 ** targ.size)

    # Because the width of the bias and target registers differ in general,
    # we need to flip the bias data bits and apply an extra -dphi
    # when the sign is negative.
    for bias_bit in bias[:-1]:
        circuit.cx(sign_bit, bias_bit)

        for targ_bit in targ:
            circuit.cp(-dphi * (2 ** targ_bit.index), targ_bit, sign_bit)
            circuit.mcx([bias_bit, targ_bit], ancilla)
            circuit.crz(-2. * dphi * (2 ** (bias_bit.index + targ_bit.index)), ancilla, sign_bit)
            circuit.mcx([bias_bit, targ_bit], ancilla)

        circuit.cx(sign_bit, bias_bit)

    if gatify < 0 or gatify >= 1:
        return to_gate(circuit, 'add_bias')
    else:
        return circuit

    
def set_target_phase(layer_input, weights, bias, targ, activation, gatify=-1):
    """Apply phases to the bases of the target register to make sum_{k} e^{2*pi*i*output*k}|k>
    
    Args:
        layer_input (list[QuantumRegister]): Values of the nodes in the previous layer.
        weights (list[QuantumRegister]): Binary weights connecting the input nodes to the target.
        bias (QuantumRegister): Bias on the target node.
        targ (QuantumRegister): Value of the target node.
        activation (bool): Whether to activate layer_input with ReLU.
        
    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """
    
    registers = flatten_list(layer_input, weights, bias, targ)
    circuit = QuantumCircuit(*registers)
    
    # Loop over nodes in the previous layer
    for source, weight in zip(layer_input, weights):
        # weight qubit: 0 -> w=-1, 1 -> w=+1
        circuit += add_weighted_input(source, weight[0], targ, activation, gatify=gatify)
        
    circuit += add_bias(bias, targ, gatify=gatify)
    
    if gatify < 0 or gatify >= 3:
        return to_gate(circuit, 'set_target_phase')
    else:
        return circuit

    
def inverse_qft(targ, gatify=-1):
    circuit = QuantumCircuit(targ)
    
    iqft = QFT(targ.size, inverse=True, do_swaps=False)
    circuit.compose(iqft, targ, inplace=True)
    for iq in range(targ.size // 2):
        circuit.swap(targ[iq], targ[-iq - 1])
        
    if gatify < 0 or gatify >= 1:
        return to_gate(circuit, 'inverse_qft')
    else:
        return circuit


def propagate_one(layer_input, weights, bias, targ, activation, gatify=-1):
    """Propagate the output of all nodes in the previous layer through weights
    to the target node.
    
    Args:
        layer_input (list[QuantumRegister]): Values of the nodes in the previous layer.
        weights (list[QuantumRegister]): Binary weights connecting the input nodes to the target.
        bias (QuantumRegister): Bias on the target node.
        targ (QuantumRegister): Value of the target node.
        activation (bool): Whether to activate layer_input with ReLU.
        
    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """

    registers = flatten_list(layer_input, weights, bias, targ)
    circuit = QuantumCircuit(*registers)
    
    # Prepare the next input register for phase estimation
    circuit.h(targ)
    
    circuit += set_target_phase(layer_input, weights, bias, targ, activation, gatify=gatify)

    circuit += inverse_qft(targ)

    if gatify < 0 or gatify >= 4:
        return to_gate(circuit, 'propagate_one')
    else:
        return circuit

    
def propagate(layer_input, layer_weights, layer_biases, layer_output, activation, gatify=-1):
    """Propagate the output of all nodes in the previous layer through weights.
    
    Args:
        layer_input (list[QuantumRegister]): Values of the nodes in the previous layer.
        layer_weights (list[list[QuantumRegister]]): Binary weights connecting the input nodes to the target.
        layer_biases (list[QuantumRegister]): Biases on the next layer.
        layer_output (list[QuantumRegister]): Values of the nodes in the next layer.
        activation (bool): Whether to activate layer_input with ReLU.
        
    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """
    
    registers = flatten_list(layer_input, layer_weights, layer_biases, layer_output)
    circuit = QuantumCircuit(*registers)
    
    for weights, bias, targ in zip(layer_weights, layer_biases, layer_output):
        circuit += propagate_one(layer_input, weights, bias, targ, activation, gatify=gatify)

    if gatify < 0 or gatify >= 5:
        return to_gate(circuit, 'propagate')
    else:
        return circuit

    
def feedforward(reg_data, reg_weights, reg_biases, gatify=-1):
    """Apply weights to the input data and compute the output of the network.
    
    Args:
        reg_data (list[list[QuantumRegister]]): Data registers.
        reg_weights (list[list[list[QuantumRegister]]]): Weight registers.
        reg_biases (list[list[QuantumRegister]]): Bias registers.

    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """
    
    registers = flatten_list(reg_data, reg_weights, reg_biases)
    circuit = QuantumCircuit(*registers)
    
    for ilayer in range(len(reg_data) - 1):
        layer_input = reg_data[ilayer]
        layer_output = reg_data[ilayer + 1]
        layer_weights = reg_weights[ilayer]
        layer_biases = reg_biases[ilayer]
        
        circuit += propagate(layer_input, layer_weights, layer_biases, layer_output, ilayer > 0, gatify=gatify)
        if gatify >= 0:
            circuit.barrier()
            
    if gatify < 0 or gatify >= 5:
        return to_gate(circuit, 'feed_forward')
    else:
        return circuit


def transduce_amplitude(y, out_node_data, amp, gatify=-1):
    """Transduce the amplitude `[cos(pi/4 - out)]` for each sample from the output node data.
    Normalizes `out` so that we work in the upper quadrant.
    
    Args:
        y (Qubit): Input y.
        out_node_data (QuantumRegister): Register containing data for the output node.
        amp (Qubit): Qubit to perform Ry rotations on.
        
    Returns:
        QuantumCircuit: Circuit corresponding to this subroutine.
    """

    circuit = QuantumCircuit(y.register, out_node_data, amp.register)

    # Flip the ancilla according to the parity between the sign of the output node data and input y
    # (o<0 && y==1) or (o>=0 && y==0) -> reduce amplitude
    sign_bit = out_node_data[-1]
    circuit.cx(sign_bit, y)

    # We are going to move the angles between 0 and pi/2, not 0 and 2pi. Need to flip the data bits
    # if sign_bit is 1
    circuit.cx(sign_bit, out_node_data[:-1])

    # Will use H-Rz-H instead of repeated Rys
    circuit.h(amp)
    circuit.rz(-np.pi / 2., amp)

    # Now y==1 implies parity==1. Flip the phase of the amp qubit using the identity
    # X-Rz(phi)-X = Rz(-phi)
    circuit.cx(y, amp)

    dtheta = np.pi / 4. / (2 ** (out_node_data.size - 1))

    # Give a dtheta kick if sign_bit is 1 because data[:-1] == 0 means -1 in that case
    circuit.crz(-2. * dtheta, sign_bit, amp)

    for idigit in range(out_node_data.size - 1):
        # We want to increase the amplitude when the parity is -1
        # -> reduce the phase when the parity is -1
        # -> reduce the phase when the phase is flipped through the cx above
        # -> increase the phase
        # -> perform Rz(-theta) = exp(+i theta/2 Z)
        circuit.crz(-2. * dtheta * (2 ** idigit), out_node_data[idigit], amp)

    circuit.cx(y, amp)

    circuit.h(amp)

    if CLEAN_UP_AFTER_ONESELF:
        circuit.cx(sign_bit, out_node_data[:-1])
        circuit.cx(sign_bit, y)

    if gatify != 0:
        return to_gate(circuit, 'transduce_amplitude')
    else:
        return circuit

# https://github.com/skushagra/quantum/blob/911161315a44563c97fdd1ee4c2ffacc5030ce78/quantum.py
def circ(a,b):
    from qiskit import QuantumCircuit
    c= QuantumCircuit(a,b)
    return c
def plot(pt,counts):
    if pt == 'hist':
        from qiskit.tools.visualization import plot_histogram
        plot_histogram(counts)
    if pt == 'bloch_mv':
        from qiskit.tools.visualization import plot_bloch_multivector
        plot_bloch_multivector(counts)
def qreg(a):
    from qiskit import QuantumRegister
    q  = QuantumRegister(a)
    return q
def creg(a):
    from qiskit import ClassicalRegister
    c = ClassicalRegister(a)
    return c
def exe(c, b, s):
    from qiskit import execute
    from qiskit import Aer
    b = Aer.get_backend(b)
    return execute(c, backend=b, shots=s).result()
def counts(r):
    return r.get_counts()
def draw(c):
    return c.draw()

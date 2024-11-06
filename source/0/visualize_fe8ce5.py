# https://github.com/ericdweise/quantum/blob/41557d4423dfdbf1d04144436e9bfffd407b95aa/qiskit/myqiskit/visualize.py
from qiskit import *
from IPython.display import display, Markdown, Latex


unitary_simulator = Aer.get_backend('unitary_simulator')
statevector_simulator = Aer.get_backend('statevector_simulator')


def complex_pretty(c):
    if (abs(c.imag - 0) < 0.01) and (abs(c.real - 0) < 0.01):
        return '0'

    elif abs(c.imag - 0) < 0.01:
        if abs(c.real - round(c.real)) < 0.01:
            return f'{int(round(c.real))}'
        else:
            return f'{c.real:.2f}'

    elif abs(c.real - 0) < 0.01:
        if abs(c.imag - round(c.imag)) < 0.01:
            return f'{int(round(c.imag))}j'
        else:
            return f'{c.imag:.2f}j'

    if abs(c.real - round(c.real)) < 0.01:
        rp = f'{int(round(c.real))}'
    else:
        rp = f'{c.real:.2f}'

    if c.imag > 0:
        op = ' + '
        if abs(c.imag - round(c.imag)) < 0.01:
            ip = f'{int(round(c.imag))}'
        else:
            ip = f'{c.imag:.2f}'

    else:
        op = ' - '
        if abs(c.imag - round(c.imag)) < 0.01:
            ip = f'{-int(c.imag)}'
        else:
            ip = f'{-c.imag:.2f}'

    return(rp + op + ip + 'j')


def get_unitary(circuit):
    job = execute(circuit, unitary_simulator)
    result = job.result()
    return result.get_unitary()

def matrix_pretty(circuit):
    gate = get_unitary(circuit)
    gate_latex = '\\begin{pmatrix}'
    for line in gate:
        for element in line:
            gate_latex += complex_pretty(element) + '&'
        gate_latex  = gate_latex[0:-1]
        gate_latex +=  '\\\\'
    gate_latex  = gate_latex[0:-2]
    gate_latex += '\end{pmatrix}'
    display(Markdown(gate_latex))


def statevector(circuit):
    job = execute(circuit, statevector_simulator)
    result = job.result()
    return result.get_statevector()


def statevector_pretty(circuit):
    svector = statevector(circuit)
    latex = '\\begin{pmatrix}'
    for element in svector:
        latex += complex_pretty(element)
        latex +=  '\\\\'
    latex  = latex[0:-2]
    latex += '\end{pmatrix}'
    display(Markdown(latex))

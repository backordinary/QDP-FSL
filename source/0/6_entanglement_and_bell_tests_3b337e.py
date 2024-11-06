# https://github.com/NTrevisani/DataScienceMaster/blob/b32ae615a4412143d6486ed767630f46ee4cfed4/QuantumComputing/6_Entanglement_and_Bell_tests.py
#!/usr/bin/env python
# coding: utf-8

# $\newcommand{\ket}[1]{\left|#1\right>}$
# 
# $\newcommand{\bra}[1]{\left<#1\right|}$
# 
# $\newcommand{\braket}[2]{\left<#1 | #2\right>}$
# 
# $\newcommand{\expectation}[1]{\left<#1\right>}$
# 

# # Entanglement and Bell tests
# 
# For this notebook, I have tried to integrate the content of IBM Q tutorial with some wikipedia examples: 
# 
# https://quantum-computing.ibm.com/docs/guide/mult-entang/entanglement-and-bell-tests <br>
# https://es.wikipedia.org/wiki/Teorema_de_Bell
# 
# One of the most interesting but at the same time counterintuitive features of quantum mechanics, is that it states that two systems which are put and kept in a coherent (entangled) state can behave in a strongly correlated way, even if they are too far apart to let any classical local theory explaining that.
# 
# Of course, this fact influences the behaviour of quantum computers, so that we can verify it and see its effect.

# ## The Bell test
# 
# To understand how entanglement prevents quantum mechanics from being a local and realist theory, we will proceed by setting up an experiment which works in classical mechanics, but fails in quantum mechanics.
# 
# It is known as the Bell test and we will present it in terms of qbits and measurement of their states, while originally we would have to deal with electrons and measurement of their spins, since it is equivalent and it keeps us in the quantum computing environment.

# ### Experimental setup
# 
# Let's suppose we prepare two qbits in a coherent state as for example:
# 
# $\ket{\psi} = \dfrac{\ket{00} + \ket{11}}{\sqrt{2}}$
# 
# If we measure the two qbits along the same *direction* (remember the Bloch sphere and the fact that the z-axis connects the $\ket{0}$ and $\ket{1}$ states), we will always get the same results for both qbits.
# In other words, if perform measurement to see if the first qbit is in the $\ket{0}$ state and we get a positive result, also the measurement of the second qbit state will give a positive result.
# This is valid not only for measurements along the z-axis, but along any axis of the Bloch sphere, the important thing is that we have to measure both the qbits along the same axis.
# 
# As a summary, the results of the measurements of the two qbits, if performed along the same axis, are fully correlated (correlation = +1).
# 
# Now, let's change a bit our measurements: the axes along which we measure the states of the two qbits will now be othogonal to each other.
# For example, if we measure the state of the first qbit along the z axis, we measure the state of the second qbit along the x axis. This is equivalent to measure if the first qbit is in the $\ket{0}$ state and if the second qbit is in the $\ket{+} = \dfrac{\ket{0} + \ket{1}}{\sqrt{2}}$ state.
# In this case, the result of the measurement of the first qbit state will not ensure anything about the state of the second qbit. So that, if we measure the first qbit to be in the $\ket{0}$ state, the second qbit may be in the $\ket{+}$ state or not, with 50% probability.
# 
# So, the results of the measurements of the two qbits, if performed along two othogonal axes, are not correlated (correlation = 0).
# 
# Up to this point, everything can be explained by the existence of hidden variables that makes it possible to have always fully correlated results when the measurement is made on the same axis for the two qbits and uncorrelated results when the measurement is performed on orthogonal axes.
# We call them *hidden* variables since we don't have direct acces to them: when we perform a measurement, we just see the result of (indirectly) setting them to the *correct* values in the prepation of the entangled state.
# 
# Since all the information needed by the qbits is, in our current assumption, stored in some hidden variable, we are now assuming quantum mechanics is a local and realist theory:
# - realism: all observables have a definite value independent of the measurement;
# - locality: no information can travel faster than the speed of light.
# 
# It is easy to understand that the results of our measurement depend (under our current assumption) on the state of the hidden variables, so that if we were able to know their state (which we can't!) we would know the outcome of the measurement *before* performing it (realism) so that the qbits do not have to *communicate* with each other to agree on the result of the measurement (locality). 

# ### The Clauser-Horne-Shimony-Holt inequality
# 
# Let's now consider another slightly different experimental setup: 
# - instead of measuring only one direction, each qbit state can be measured along two orthogonal directions (e.g. the first qbit can be measured along z-axis or along x-axis, so we can measure if it is in the $\ket{0}$ or if it is in the $\ket{+}$ state);
# - the set of axes along which we measure two different qbits are rotated by 45$^{\circ}$ with respect to each other. So, if the two axes we choose for measuring the first qbits are the z-axis (vertical axis) and the x-axis (horizontal axis), the two axes we use for measuring the second qbits are the ones shown in the following Figure.

# <img src='Bell.png'>

# What we want to do now is to choose one of the two axes for the first qbit measurement, one of the two axes for the second qbit measurement, and measure their states along those axes.
# 
# Now let's define a set of scores:
# - $\expectation{AB}$: measurement of 1st qbit along z-axis ($A$) and second qbit along 45$^{\circ}$-axis ($B$) is the same: +1 (-1 if different);
# - $\expectation{AB'}$: measurement of 1st qbit along z-axis ($A$) and second qbit along 135$^{\circ}$-axis ($B'$) is the same: +1 (-1 if different);
# - $\expectation{A'B}$: measurement of 1st qbit along x-axis ($A'$) and second qbit along 45$^{\circ}$-axis ($B$) is the same: +1 (-1 if different);
# - $\expectation{A'B'}$: measurement of 1st qbit along x-axis ($A'$) and second qbit along 135$^{\circ}$-axis ($B'$) is the same: +1 (-1 if different).
# 
# Clauser, Horne, Shimony, and Holt (CHSH) demonstrated that assuming realism and locality, then the following inequality must be true:
# 
# $ |C| = |\expectation{AB} + \expectation{AB'} + \expectation{A'B} - \expectation{A'B'}| \leq 2$
# 
# Let's verify it in our case combining all the possible outcomes of our measurements, again considering that some hidden variables have been given some particular values when we created the entanglement between our two qbits, so that realism and locality are satisfied.
# 
# In the following tables, we are going to list all the possible results we can get, and the corresponding values of C. 

# | Hidden variables states             |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
# |:------------------------------------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
# | $A$: z-axis (1st qbit)              | + | + | + | + | − | − | − | − | + | + | + | + | − | − | − | − |
# | $B$: 45$^{\circ}$-axis (2nd qbit)   | + | + | + | − | − | − | − | + | + | − | − | − | + | + | + | − |
# | $A'$: x-axis (1st qbit)             | + | + | − | − | − | − | + | + | − | + | + | − | + | − | − | + |
# | $B'$: 135$^{\circ}$-axis (2nd qbit) | + | − | − | − | − | + | + | + | + | + | − | + | − | + | − | − |

# | Score                 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
# |:----------------------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
# | $\expectation{AB}$    | +1 | +1 | +1 | −1 | +1 | +1 | +1 | -1 | +1 | −1 | −1 | −1 | −1 | −1 | −1 | +1 |
# | $\expectation{A'B}$   | +1 | +1 | −1 | +1 | +1 | +1 | −1 | +1 | −1 | −1 | −1 | +1 | +1 | −1 | −1 | −1 |
# | −$\expectation{A'B'}$ | −1 | +1 | −1 | −1 | −1 | +1 | −1 | −1 | +1 | −1 | +1 | +1 | +1 | +1 | −1 | +1 |
# | $\expectation{AB'}$   | +1 | −1 | −1 | −1 | +1 | −1 | −1 | −1 | +1 | +1 | −1 | +1 | +1 | −1 | +1 | +1 |
# | $C$                   | +2 | +2 | −2 | −2 | +2 | +2 | −2 | −2 | +2 | −2 | −2 | +2 | +2 | −2 | −2 | +2 |

# So, everything looks ok: we assumed that the strange results of entangled systems can be explained by the existence of hidden variables that, whatever they are, makes it possible to satisfy the CHSH inequality and respect locality and realism.

# ### Quantum mechanical approach
# 
# Now, let's try to solve the same problem using quantum mechanics.
# 
# As we know, in quantum mechanics the measurement process has an active role, as the act of measuring the state of a system makes it collapse to one of the possible final state, with a probability equal to the square module of the corresponding coefficient. In case the results of quantum mechanics are different from those we got in the previous section, it means that quantum mechanics is not a realist or a local theory.
# 
# In that case, we can use our quantum computer to test which one of the two approaches to the solution of our problem better adjusts to reality. 
# This highlights an interesting features of quantum computers: they represent unique tools to test quantum mechanics.
# 
# It is worth at this point taking some time to understand the meaning of the scores defined earlier.
# Each of the score represent the correlation between the two measurements and in fact, it takes value +1 when the measurements give the same result (so they are fully correlated) and it takes value -1 when the results are opposite (and they are fully uncorrelated).
# 
# In quantum mechanics, we can get the expected correlation between the measurements of two entangled qbits by calculating the expectation value $\bra{\psi} A \ket{\psi}$ of a measurement.

# So let's start preparing everything we need to solve the problem.
# 
# First of all, our two qbits, as already said, are in the following entangled state:
# 
# $\ket{\psi} = \dfrac{\ket{00} + \ket{11}}{\sqrt{2}}$
# 
# For the first qbit, we have two possibilities:
# - measure its state along the z-axis or, equivalently, measure if it is in the $\ket{0}$ state;
# - measure its state along the x-axis or, equivalently, measure if it is in the $\ket{+}$ state.
# 
# So, we recall the matrix expression of the Z and X operators:
# 
# $ Z =   
# \begin{bmatrix}
#     1 & 0 \\
#     0 & -1 
# \end{bmatrix}
# $
# 
# $ X =   
# \begin{bmatrix}
#     0 & 1 \\
#     1 & 0 
# \end{bmatrix} 
# $

# While for the second qbit, the two axes along which we want to perform the measurements are rotated by 45$^{\circ}$ with respect to the axes of the first qbit. The operators we need for these measurements can be written as:
# 
# $ W = \dfrac{Z + X}{\sqrt{2}} =   
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#     1 & 1 \\
#     1 & -1 
# \end{bmatrix}
# $
# 
# $ V = \dfrac{Z - X}{\sqrt{2}} =   
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#     1 & -1 \\
#     -1 & -1 
# \end{bmatrix}
# $
# 
# Remembering that each operator acts only on the corresponding qbit, independently on the other one, we can list all the possible results we can get:

# #### measurement of the state of the first qbit along the z-axis
# 
# $Z \ket{0} = 
# \begin{bmatrix}
#     1 & 0 \\
#     0 & -1 
# \end{bmatrix}
# \begin{bmatrix}
#     1 \\
#     0 
# \end{bmatrix} = 
# \begin{bmatrix}
#     1 \\
#     0 
# \end{bmatrix} = 
# \ket{0}
# $
# 
# $Z \ket{1} = 
# \begin{bmatrix}
#     1 & 0 \\
#     0 & -1 
# \end{bmatrix}
# \begin{bmatrix}
#     0 \\
#     1 
# \end{bmatrix} =
# \begin{bmatrix}
#     0 \\
#     -1 
# \end{bmatrix} = 
# - \ket{1}
# $

# #### measurement of the state of the first qbit along the x-axis
# 
# $X \ket{0} = 
# \begin{bmatrix}
#     0 & 1 \\
#     1 & 0 
# \end{bmatrix}
# \begin{bmatrix}
#     1 \\
#     0 
# \end{bmatrix} = 
# \begin{bmatrix}
#     0 \\
#     1 
# \end{bmatrix} = 
# \ket{1}
# $
# 
# $X \ket{1} = 
# \begin{bmatrix}
#     0 & 1 \\
#     1 & 0 
# \end{bmatrix}
# \begin{bmatrix}
#     0 \\
#     1 
# \end{bmatrix} =
# \begin{bmatrix}
#     1 \\
#     0 
# \end{bmatrix} = 
# \ket{0}
# $

# #### measurement of the state of the second qbit along the w-axis
# 
# $W \ket{0} = 
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#     1 & 1 \\
#     1 & -1 
# \end{bmatrix}
# \begin{bmatrix}
#     1 \\
#     0 
# \end{bmatrix} = 
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#     1 \\
#     1 
# \end{bmatrix} = 
# \ket{+}
# $
# 
# $W \ket{1} = 
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#     1 & 1 \\
#     1 & -1 
# \end{bmatrix}
# \begin{bmatrix}
#     0 \\
#     1 
# \end{bmatrix} = 
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#     1 \\
#     -1 
# \end{bmatrix} = 
# \ket{-}
# $

# #### measurement of the state of the second qbit along the v-axis
# 
# $V \ket{0} = 
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#      1 & -1 \\
#     -1 & -1 
# \end{bmatrix}
# \begin{bmatrix}
#     1 \\
#     0 
# \end{bmatrix} = 
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#     1 \\
#     -1 
# \end{bmatrix} = 
# \ket{-}
# $
# 
# $V \ket{1} = 
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#      1 & -1 \\
#     -1 & -1 
# \end{bmatrix}
# \begin{bmatrix}
#     0 \\
#     1 
# \end{bmatrix} = 
# \dfrac{1}{\sqrt{2}}
# \begin{bmatrix}
#     -1 \\
#     -1 
# \end{bmatrix} = 
# -\ket{+}
# $

# ### Getting the scores
# 
# Before moving to the scores, let's change a bit the notation for the two-qbits state and make clearer, for every state, to which qbit it refers:
# 
# $\ket{\psi} = \dfrac{\ket{00} + \ket{11}}{\sqrt{2}} = 
# \dfrac{\ket{0}_1 \otimes \ket{0}_2 + \ket{1}_1 \otimes \ket{1}_2}{\sqrt{2}}
# $
# 
# where the substript indicates to which qbit the state refers to and the $\otimes$ symbol underlines the fact that each operator acts only on the state associated to its qbit, leaving the state of the other untouched.

# #### $\expectation{AB}$ score
# 
# $\expectation{AB} = \bra{\psi} Z_1 W_2 \ket{\psi} =
# %
# \bra{\psi} \dfrac{Z_1 \ket{0}_1 \otimes W_2 \ket{0}_2 + Z_1 \ket{1}_1 \otimes W_2 \ket{1}_2}{\sqrt{2}} = 
# %
# \bra{\psi} \dfrac{\ket{0}_1 \otimes \ket{+}_2 + (-\ket{1}_1) \otimes \ket{-}_2}{\sqrt{2}} = 
# %
# \bra{\psi} 
# \dfrac{\ket{0}_1 \otimes
# \left( \dfrac{\ket{0}_2 + \ket{1}_2}{\sqrt{2}} \right) - 
# \ket{1}_1 \otimes 
# \left( \dfrac{\ket{0}_2 - \ket{1}_2}{\sqrt{2}} \right)}
# {\sqrt{2}} = 
# %
# \bra{\psi} 
# \dfrac{\ket{0}_1 \otimes \ket{0}_2 + 
# \ket{0}_1 \otimes \ket{1}_2 - 
# \ket{1}_1 \otimes \ket{0}_2 + 
# \ket{1}_1 \otimes \ket{1}_2}
# {2} = 
# %
# \bra{\psi} 
# \dfrac{\ket{00} + \ket{01} - \ket{10} + \ket{11}}{2} = 
# %
# \left[
# \dfrac{\bra{00} + \bra{11}}{\sqrt{2}} 
# \right]
# \left[
# \dfrac{\ket{00} + \ket{01} - \ket{10} + \ket{11}}{2} 
# \right] = \dfrac{1 + 1}{2\sqrt{2}} = \dfrac{1}{\sqrt{2}}
# $

# #### $\expectation{AB'}$ score
# 
# $\expectation{AB'} = \bra{\psi} Z_1 V_2 \ket{\psi} =
# %
# \bra{\psi} \dfrac{Z_1 \ket{0}_1 \otimes V_2 \ket{0}_2 + Z_1 \ket{1}_1 \otimes V_2 \ket{1}_2}{\sqrt{2}} = 
# %
# \bra{\psi} \dfrac{\ket{0}_1 \otimes \ket{-}_2 + (-\ket{1}_1) \otimes (-\ket{+}_2)}{\sqrt{2}} = 
# %
# \bra{\psi} 
# \dfrac{\ket{0}_1 \otimes
# \left( \dfrac{\ket{0}_2 - \ket{1}_2}{\sqrt{2}} \right) + 
# \ket{1}_1 \otimes 
# \left( \dfrac{\ket{0}_2 + \ket{1}_2}{\sqrt{2}} \right)}
# {\sqrt{2}} = 
# %
# \bra{\psi} 
# \dfrac{\ket{0}_1 \otimes \ket{0}_2 - 
# \ket{0}_1 \otimes \ket{1}_2 + 
# \ket{1}_1 \otimes \ket{0}_2 + 
# \ket{1}_1 \otimes \ket{1}_2}
# {2} = 
# %
# \bra{\psi} 
# \dfrac{\ket{00} - \ket{01} + \ket{10} + \ket{11}}{2} = 
# %
# \left[
# \dfrac{\bra{00} + \bra{11}}{\sqrt{2}} 
# \right]
# \left[
# \dfrac{\ket{00} - \ket{01} + \ket{10} + \ket{11}}{2} 
# \right] = \dfrac{1 + 1}{2\sqrt{2}} = \dfrac{1}{\sqrt{2}}
# $

# #### $\expectation{A'B}$ score
# 
# $\expectation{A'B} = \bra{\psi} X_1 W_2 \ket{\psi} =
# %
# \bra{\psi} \dfrac{X_1 \ket{0}_1 \otimes W_2 \ket{0}_2 + X_1 \ket{1}_1 \otimes W_2 \ket{1}_2}{\sqrt{2}} = 
# %
# \bra{\psi} \dfrac{\ket{1}_1 \otimes \ket{+}_2 + \ket{0}_1 \otimes \ket{-}_2}{\sqrt{2}} = 
# %
# \bra{\psi} 
# \dfrac{\ket{1}_1 \otimes
# \left( \dfrac{\ket{0}_2 + \ket{1}_2}{\sqrt{2}} \right) + 
# \ket{0}_1 \otimes 
# \left( \dfrac{\ket{0}_2 - \ket{1}_2}{\sqrt{2}} \right)}
# {\sqrt{2}} = 
# %
# \bra{\psi} 
# \dfrac{\ket{1}_1 \otimes \ket{0}_2 + 
# \ket{1}_1 \otimes \ket{1}_2 + 
# \ket{0}_1 \otimes \ket{0}_2 - 
# \ket{0}_1 \otimes \ket{1}_2}
# {2} = 
# %
# \bra{\psi} 
# \dfrac{\ket{10} + \ket{11} + \ket{00} - \ket{01}}{2} = 
# %
# \left[
# \dfrac{\bra{00} + \bra{11}}{\sqrt{2}} 
# \right]
# \left[
# \dfrac{\ket{10} + \ket{11} + \ket{00} - \ket{01}}{2} 
# \right] = \dfrac{1 + 1}{2\sqrt{2}} = \dfrac{1}{\sqrt{2}}
# $

# #### $\expectation{A'B'}$ score
# 
# $\expectation{A'B'} = \bra{\psi} X_1 V_2 \ket{\psi} =
# %
# \bra{\psi} \dfrac{X_1 \ket{0}_1 \otimes V_2 \ket{0}_2 + X_1 \ket{1}_1 \otimes V_2 \ket{1}_2}{\sqrt{2}} = 
# %
# \bra{\psi} \dfrac{\ket{1}_1 \otimes \ket{-}_2 + \ket{0}_1 \otimes (- \ket{+}_2)}{\sqrt{2}} = 
# %
# \bra{\psi} 
# \dfrac{\ket{1}_1 \otimes
# \left( \dfrac{\ket{0}_2 - \ket{1}_2}{\sqrt{2}} \right) + 
# \ket{0}_1 \otimes 
# \left( -\dfrac{\ket{0}_2 + \ket{1}_2}{\sqrt{2}} \right)}
# {\sqrt{2}} = 
# %
# \bra{\psi} 
# \dfrac{\ket{1}_1 \otimes \ket{0}_2 - 
# \ket{1}_1 \otimes \ket{1}_2 - 
# \ket{0}_1 \otimes \ket{0}_2 - 
# \ket{0}_1 \otimes \ket{1}_2}
# {2} = 
# %
# \bra{\psi} 
# \dfrac{\ket{10} - \ket{11} - \ket{00} - \ket{01}}{2} = 
# %
# \left[
# \dfrac{\bra{00} + \bra{11}}{\sqrt{2}} 
# \right]
# \left[
# \dfrac{\ket{10} - \ket{11} - \ket{00} - \ket{01}}{2} 
# \right] = \dfrac{-1 -1}{2\sqrt{2}} = -\dfrac{1}{\sqrt{2}}
# $

# ### Finally, the value of CHSH inequality
# 
# If we now recall the definition of the CHSH inequality, we see:
# 
# $ 
# |C| = |\expectation{AB} + \expectation{AB'} + \expectation{A'B} - \expectation{A'B'}| =
# | \dfrac{1}{\sqrt{2}} + \dfrac{1}{\sqrt{2}} + \dfrac{1}{\sqrt{2}} - \left(- \dfrac{1}{\sqrt{2}}\right)| = 
# \dfrac{4}{\sqrt{2}} = 2 \sqrt{2} > 2
# $
# 
# Which means that quantum mechanics does not respect the requirements needed by a theory to be realist and local.
# This means that, if with our quantum computer agrees with quantum mechanics, we have to accept that no hidden variables can explain our results, that the act of measuring the state of our qbits make them *choose together* the outcome.  

# ### Experimental verification
# 
# We can finally move to the experimental verification of our assumptions:
# 1. Quantum mechanics is a local and realist theory and the peculiar behaviour of entangled state can be explained by the existence of hidden variables. In this case, $|C| \leq 2$;
# 2. Quantum mechanics is not local nor realist, the measurement of the state of an entangled system has an active effect on it, making the qbits to choose *together* the outcome. In this case, $|C| \leq 2\sqrt{2}$.  
# 
# To do it, we will have to prepare a quantum circuit in the so-called Bell state: 
# 
# $\ket{\psi} = \dfrac{\ket{00} + \ket{11}}{\sqrt{2}}$
# 
# And then perform all the measurements along the pairs of axes we defined in the previous sections of the notebook.
# By properly summing the results of each measurement, we will get the value of $C$.
# 
# To prepare the Bell state, we will use a $H$ gate to put the first qbit in the 
# $\ket{+} = \frac{\ket{0} + \ket{1}}{\sqrt{2}}$ 
# superposition state and then we will apply a $CNOT$ gate to flip the state of the second qbit from $\ket{0}$ to $\ket{1}$ when the first qbit is in state $\ket{1}$.
# 
# Let's verify here if this works on a real quantum computer as backend.

# In[1]:


# Build the circuit

from qiskit import QuantumCircuit, execute, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, Aer

# Get access to IBM Q backend
provider = IBMQ.load_account()

# Define the Quantum and Classical Registers
q = QuantumRegister(2)
c = ClassicalRegister(2)

# Create the circuit
bell_state_measurement = QuantumCircuit(q, c)

# Put the circuit in the Bell state |00> + |11>
bell_state_measurement.h(0)
bell_state_measurement.cnot(0,1)

# Measure the state
bell_state_measurement.measure(q, c)


# In[2]:


# Draw the circuit
bell_state_measurement.draw(output='mpl')


# In[3]:


# Execute the circuit

backend = provider.get_backend('ibmq_rome')

job = execute(bell_state_measurement, 
              backend = backend, 
              shots=1024)

result = job.result()


# In[4]:


# Print the result
print(result.get_counts(bell_state_measurement))


# In[5]:


# Plot the result
from qiskit.visualization import plot_histogram

plot_histogram(result.get_counts(bell_state_measurement))


# ### $\expectation{AB}$ measurement

# In[6]:


### Build the circuit

from qiskit import QuantumCircuit, execute, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, Aer
import numpy as np

# Get access to IBM Q backend
provider = IBMQ.load_account()

# Define the Quantum and Classical Registers
q = QuantumRegister(2)
c = ClassicalRegister(2)

# Create the circuit
ab_measurement = QuantumCircuit(q, c)

# Put the circuit in the Bell state |00> + |11>
ab_measurement.h(0)
ab_measurement.cnot(0,1)

# I want to measure the second qbit along W = X + Z.
# But I can measure it only along Z.
# I rotate the state of the second qbit of the same angle
# needed to rotate X + Z on Z --> theta = 45 degrees
ab_measurement.u3(-np.pi/4, 0, 0, q[1])

# Measure the state
ab_measurement.measure(q, c)


# In[7]:


# Draw the circuit
ab_measurement.draw(output='mpl')


# In[8]:


# Execute the circuit

backend = provider.get_backend('ibmq_rome')
shots = 1024

job = execute(ab_measurement, 
              backend = backend, 
              shots=shots)

ab_result = job.result()


# In[9]:


# Print the result
print(ab_result.get_counts(ab_measurement))


# In[10]:


# Plot the result
from qiskit.visualization import plot_histogram

plot_histogram(ab_result.get_counts(ab_measurement))


# In[11]:


# Get the correlation value
def correlator(result, circuit, shots):
    my_dict = result.get_counts(circuit)
    return (my_dict['00'] + my_dict['11'] - my_dict['10'] - my_dict['01']) / shots     


# In[12]:


ab_correlation = correlator(ab_result, ab_measurement, shots)
ab_correlation


# ### $\expectation{A'B}$ measurement

# In[13]:


### Build the circuit

from qiskit import QuantumCircuit, execute, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, Aer
import numpy as np

# Get access to IBM Q backend
provider = IBMQ.load_account()

# Define the Quantum and Classical Registers
q = QuantumRegister(2)
c = ClassicalRegister(2)

# Create the circuit
apb_measurement = QuantumCircuit(q, c)

# Put the circuit in the Bell state |00> + |11>
apb_measurement.h(0)
apb_measurement.cnot(0,1)

# I want to measure the first qbit along X.
# But I can measure it only along Z.
# I apply a H gate to the first qbit,
# to rotate it from X toward Z.
apb_measurement.h(0)

# I want to measure the second qbit along W = X + Z.
# But I can measure it only along Z.
# I rotate the state of the second qbit of the same angle
# needed to rotate X + Z on Z --> theta = 45 degrees
apb_measurement.u3(-np.pi/4, 0, 0, q[1])

# Measure the state
apb_measurement.measure(q, c)


# In[14]:


# Draw the circuit
apb_measurement.draw(output='mpl')


# In[15]:


# Execute the circuit

backend = provider.get_backend('ibmq_rome')
shots = 1024

job = execute(apb_measurement, 
              backend = backend, 
              shots=shots)

apb_result = job.result()


# In[16]:


# Print the result
print(apb_result.get_counts(apb_measurement))


# In[17]:


# Plot the result
from qiskit.visualization import plot_histogram

plot_histogram(apb_result.get_counts(apb_measurement))


# In[18]:


apb_correlation = correlator(apb_result, apb_measurement, shots)
apb_correlation


# ### $\expectation{AB'}$ measurement

# In[19]:


### Build the circuit

from qiskit import QuantumCircuit, execute, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, Aer
import numpy as np

# Get access to IBM Q backend
provider = IBMQ.load_account()

# Define the Quantum and Classical Registers
q = QuantumRegister(2)
c = ClassicalRegister(2)

# Create the circuit
abp_measurement = QuantumCircuit(q, c)

# Put the circuit in the Bell state |00> + |11>
abp_measurement.h(0)
abp_measurement.cnot(0,1)

# I want to measure the second qbit along V = Z - X.
# But I can measure it only along Z.
# I rotate the state of the second qbit of the same angle
# needed to rotate Z - X on Z --> theta = 45 degrees
abp_measurement.u3(np.pi/4, 0, 0, q[1])

# Measure the state
abp_measurement.measure(q, c)


# In[20]:


# Draw the circuit
abp_measurement.draw(output='mpl')


# In[21]:


# Execute the circuit

backend = provider.get_backend('ibmq_rome')
shots = 1024

job = execute(abp_measurement, 
              backend = backend, 
              shots=shots)

abp_result = job.result()


# In[22]:


# Print the result
print(abp_result.get_counts(abp_measurement))


# In[23]:


# Plot the result
from qiskit.visualization import plot_histogram

plot_histogram(abp_result.get_counts(abp_measurement))


# In[24]:


abp_correlation = correlator(abp_result, abp_measurement, shots)
abp_correlation


# ### $\expectation{A'B'}$ measurement

# In[25]:


### Build the circuit

from qiskit import QuantumCircuit, execute, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, Aer
import numpy as np

# Get access to IBM Q backend
provider = IBMQ.load_account()

# Define the Quantum and Classical Registers
q = QuantumRegister(2)
c = ClassicalRegister(2)

# Create the circuit
apbp_measurement = QuantumCircuit(q, c)

# Put the circuit in the Bell state |00> + |11>
apbp_measurement.h(0)
apbp_measurement.cnot(0,1)

# I want to measure the first qbit along X.
# But I can measure it only along Z.
# I apply a H gate to the first qbit,
# to rotate it from X toward Z.
apbp_measurement.h(0)

# I want to measure the second qbit along V = Z - X.
# But I can measure it only along Z.
# I rotate the state of the second qbit of the same angle
# needed to rotate Z - X on Z --> theta = 45 degrees
apbp_measurement.u3(np.pi/4, 0, 0, q[1])

# Measure the state
apbp_measurement.measure(q, c)


# In[26]:


# Draw the circuit
apbp_measurement.draw(output='mpl')


# In[27]:


# Execute the circuit

backend = provider.get_backend('ibmq_rome')
shots = 1024

job = execute(apbp_measurement, 
              backend = backend, 
              shots=shots)

apbp_result = job.result()


# In[28]:


# Print the result
print(apbp_result.get_counts(apbp_measurement))


# In[29]:


# Plot the result
from qiskit.visualization import plot_histogram

plot_histogram(apbp_result.get_counts(apbp_measurement))


# In[30]:


apbp_correlation = correlator(apbp_result, apbp_measurement, shots)
apbp_correlation


# ### Put everything together
# 
# We can now compute $|C|$ the value, using the results just obtained, remembering that:
# 
# $ |C| = | \expectation{AB} + \expectation{A'B} + \expectation{AB'} - \expectation{A'B'}| $

# In[33]:


# Properly sum the correlations

C = ab_correlation + apb_correlation + abp_correlation - apbp_correlation
print("The C correlation value obtained is:", C)


# Given the result we have just obtained, we can thus conclude that microscopic sytems that can be described by quantum mechanics are actually non-local and non-realist.

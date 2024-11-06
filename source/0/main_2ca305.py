# https://github.com/joinakd/Quantum_Coding_Projects/blob/e603a3c6cc52943896288064b0ed4f3723b8f270/Quantum_Games/main.py
import qiskit
from qiskit import QuantumCircuit,Aer
import numpy as np
import streamlit as st

def quantum_superposition():
    
    circuit = QuantumCircuit(1,1)
    
    circuit.h(0)

    circuit.measure(0,0)

    simulator = Aer.get_backend('aer_simulator')

    result = simulator.run(circuit).result().get_counts()

    return result

def get_random_value():
    res = quantum_superposition()

    values = list(res.values())
    keys =  list(res.keys())

    random_value = '|' + str(keys[np.argmax(values)]) + '>'
    return random_value

def validate(arr):
    """
    The Method checks if the game is finished!
    Parameters:
    arr (numpy array) : The array that serves as the board
    Returns:
    returns 0 if any of the winning condition is satisfied by any of the player else returns 1
    """

    #define a boolean variable
    flag = True
    zero_ket= '|0>'
    one_ket= '|1>'

    #Check for the principal diagonal condition w.r.t user
    if arr[0,0]==one_ket and arr[1,1]==one_ket and arr[2,2]==one_ket:
        st.success('User has won!')
        flag = False

    #Check for the principal diagonal condition w.r.t computer
    elif arr[0,0]==zero_ket and arr[1,1]==zero_ket and arr[2,2]==zero_ket:
        st.success('Computer has won!')
        flag = False

    #Check for the principal diagonal condition w.r.t user
    elif arr[0,2]==one_ket and arr[1,1]==one_ket and arr[2,0]==one_ket:
        st.success('User has won!')
        flag = False

    #Check for the principal diagonal condition w.r.t computer
    elif arr[0,2]==zero_ket and arr[1,1]==zero_ket and arr[2,0]==zero_ket:
        st.success('Computer has won!')
        flag = False

    if not flag:
        return 0

    # we execute the below for loop iff any of the above conditions are not satisfied

    if flag:

        # Checks if any of the row conquered by user
        for index in [0,1,2]:
            if list(arr[index])==[one_ket,one_ket,one_ket]:
                st.success("User has won!")
                return 0
        
        # Checks if any of the row conquered by computer
        for index in [0,1,2]:
            if list(arr[index])==[zero_ket,zero_ket,zero_ket]:
                st.success("Computer has won!")
                return 0

        # Checks if any of the column conquered by user
        for index in [0,1,2]:
            if list(arr[:,index])==[one_ket,one_ket,one_ket]:
                st.success("User has won!")
                return 0
        
        # Checks if any of the column conquered by computer
        for index in [0,1,2]:
            if list(arr[:,index])==[zero_ket,zero_ket,zero_ket]:
                st.success("Computer has won!")
                return 0
        
        # Check if it's a draw
        if '|Ψ>' not in arr:
            st.write("It's a draw!")
            return 0
        
    # if none of the above is satisfied , 1 is returned
    return 1
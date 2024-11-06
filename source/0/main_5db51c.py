# https://github.com/AmDeep/QOSF_Tasks_Code/blob/a7cedd408f8e8ef67f0c25b0f9aca9499b494017/Task1Tests/main.py
"""Generalized sum subset finder via Draper method.
"""
from itertools import combinations
from typing import Tuple

from qiskit import QuantumCircuit

from draper_adder.draper_adder import draper_adder


def draper_adder_wrapper(comb: Tuple) -> str:
    """Wrapper for draper_adder function recurisvely adds all elements in tuple

    Returns:
        str: The binary string of the sum of the elements in the tuple
    """
    if len(comb) == 0:
        return "0"
    elif len(comb) == 1:
        return comb[0]
    elif len(comb) == 2:
        return draper_adder(comb[0], comb[1])
    else:
        # do a draper adder on the first then recurse
        return draper_adder(draper_adder_wrapper(comb[1:]), comb[0])


def find_comb_sum(sub_list: list[str]) -> tuple:
    """Finds the sum of all combinations of the sublist

    Returns:
        tuple: The tuple with the combination and summation result
    """
    comb_list = []
    draper_sum_list = []
    for idx in range(len(sub_list) + 1):
        for comb in combinations(sub_list, idx):
            comb_list.append(comb)
            draper_sum_list.append(draper_adder_wrapper(comb))
    return comb_list, draper_sum_list


def subset_finder(
    int_list: list[int] = [5, 7, 8, 9, 1], output: int = 16
) -> list[(QuantumCircuit, Tuple[str])]:
    """Finds the subset of int_list that adds up to output

    Raises:
        ValueError: int_list contains negative values

    Returns:
        list: list of tuples with the circuit and the elements elements which sum to the output in a tuple
"""
    binary_list = ["{0:b}".format(num) for num in int_list]
    results = []
    if any(x < 0 for x in int_list):
        raise ValueError("Draper adder only works for positive integers")
    # Split the int_list into two parts
    list_len = len(binary_list)
    sub_list_1, sub_list_2 = binary_list[list_len // 2 :], binary_list[: list_len // 2]
    comb_list_1, sum_1 = find_comb_sum(sub_list_1)
    comb_list_2, sum_2 = find_comb_sum(sub_list_2)
    # Find the sum of the two sublists

    for i, elem_1 in enumerate(sum_1):
        for j, elem_2 in enumerate(sum_2):
            if draper_adder(elem_1, elem_2) == "{0:b}".format(output):
                valid_subset = comb_list_1[i] + comb_list_2[j]
                circ = QuantumCircuit(list_len)
                for idx, elem in enumerate(binary_list):
                    if elem in valid_subset:
                        circ.x(idx)
                results.append((circ, valid_subset))
    return results

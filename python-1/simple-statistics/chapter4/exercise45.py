import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

# Q39
q = """
The prior probabilities for events A1 and A2 are P(A1) = .40 and P(A2) + .60. It is also
known that P(A1 ^ A2) = 0. Suppose P(B | A1) = .20 and P(B | A2) = .05.
a. Are A1 and A2 mutually exclusive? Explain.
b. Compute P(A1 ^ B) and P(A2 ^ B).
c. Compute P(B).
d. Apply Bayes’ theorem to compute P(A1 | B) and P(A2 | B).
"""
heading(39)
p_a1 = 0.4
p_a2 = 0.6
p_a1_a2 = 0
p_b_a1 = 0.2
p_b_a2 = 0.05
p_b = p_b_a1 * p_a1 + p_b_a2 * p_a2
p_a1_b = p_b_a1 * p_a1 / p_b
p_a2_b = p_b_a2 * p_a2 / p_b
print(f'P(A1 ^ B) = {p_a1_b:.2f}')
print(f'P(A2 ^ B) = {p_a2_b:.2f}')
print(f'P(B) = {p_b:.2f}')
print(f'P(A1 | B) = {p_a1_b:.2f}')
print(f'P(A2 | B) = {p_a2_b:.2f}')

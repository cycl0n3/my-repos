import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

# Q52
q = """
Consider the following data and corresponding weights.
xi: 3.2 2.0 2.5 5.0
Weight (wi): 6 3 2 8  
a. Compute the weighted mean.
b. Compute the sample mean of the four data values without weighting. Note the differ-
ence in the results provided by the two computations.
"""
heading(52)
xi = np.array([3.2, 2.0, 2.5, 5.0], dtype=np.float32)
wi = np.array([6, 3, 2, 8], dtype=np.float32)
df = pd.DataFrame({'x': xi, 'w': wi})
print(df)
print(f'Weighted mean = {np.average(xi, weights=wi):.2f}')
print(f'Sample mean = {np.mean(xi):.2f}')

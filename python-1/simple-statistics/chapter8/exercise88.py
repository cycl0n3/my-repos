import math
import numpy as np
import pandas as pd

import random

from scipy import stats

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Topic 8.8: One-sided confidence intervals')

q = """E-8.12 A corporation plans to issue some short-term notes and is hoping that the interest it
will have to pay will not exceed 11.5%. To obtain some information about this prob-
lem, the corporation marketed 40 notes, one through each of 40 brokerage firms. The
mean and standard deviation for the 40 interest rates were 10.3% and .31%, respec-
tively. Since the corporation is interested in only an upper limit on the interest rates,
find a 95% upper confidence bound for the mean interest rate that the corporation will
have to pay for the notes.
"""
heading('E-8.12')
print(q)

x = 10.3
s = 0.31
n = 40
alpha = 0.05

pe = x
se = s/math.sqrt(n)
z = stats.norm.ppf(1 - alpha)
ucb = pe + z*se

print(f'Upper confidence bound: {ucb}')

q = """8.68 Suppose you wish to estimate a population mean based on a random sample of n observations, and prior
experience suggests that s = 12.7. If you wish to esti-mate m correct to within 1.6, with probability equal to
.95, how many observations should be included in your sample?
"""
heading('8.68')
print(q)

s = 12.7
e = 1.6
alpha = 0.05
z = stats.norm.ppf(1 - alpha/2)

n = (z*s/e)**2
print(f'n: {n}')

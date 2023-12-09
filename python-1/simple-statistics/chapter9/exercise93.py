import math
import numpy as np
import pandas as pd

import random

from scipy import stats

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Topic 9.3 A Large-Sample Test for the Population Mean')

q = """E-9.4 The average weekly earnings for female social workers is $670. 
Do men in the same positions have average weekly earnings that are higher than those for women? 
A random sample of n = 40 male social workers showed xbar = $725 and s = $102. 
Test the appropriate hypothesis using alpha = .01.
"""
heading('E-9.4')
print(q)

xbar = 725
s = 102
n = 40
alpha = 0.01
mu0 = 670

pe = xbar
se = s/math.sqrt(n)
z = (pe - mu0)/se
p = stats.norm.cdf(z)
print(f'z: {z}')
print(f'p: {p}')
print(f'alpha: {alpha}')

q = """E-9.5 The daily yield for a local chemical plant has averaged 880 tons for the last several
years. The quality control manager would like to know whether this average has
changed in recent months. She randomly selects 50 days from the computer database
and computes the average and standard deviation of the n = 50 yields as xbar = 871 tons 
and s = 21 tons, respectively. Test the appropriate hypothesis using alpha = .05.
"""
heading('E-9.5')
print(q)

xbar = 871
s = 21
n = 50
alpha = 0.05
mu0 = 880

pe = xbar
se = s/math.sqrt(n)
z = (pe - mu0)/se
p = stats.norm.cdf(z)

print(f'z: {z}')
print(f'p: {p}')
print(f'alpha: {alpha}')

q = """Standards set by government agencies indicate that Americans should not exceed an
average daily sodium intake of 3300 milligrams (mg). To find out whether Americans
are exceeding this limit, a sample of 100 Americans is selected, and the mean and
standard deviation of daily sodium intake are found to be 3400 mg and 1100 mg, re-
spectively. Use alpha = .05 to conduct a test of hypothesis.
"""
heading('E-9.7')
print(q)

xbar = 3400
s = 1100
n = 100
alpha = 0.05
mu0 = 3300

pe = xbar
se = s/math.sqrt(n)
z = (pe - mu0)/se
p = stats.norm.cdf(z)
p_value = 1 - p

print(f'z: {z}')
print(f'p: {p}')
print(f'p_value: {p_value}')

q = """E-9.8 The daily yield for a local chemical plant has averaged 880 tons for the last several
years. The quality control manager would like to know whether this average has
changed in recent months. She randomly selects 50 days from the computer database
and computes the average and standard deviation of the n = 50 yields as xbar = 871 tons 
and s = 21 tons, respectively. Alpha = .05.
Calculate beta for the following alternative hypotheses: H1: mu = 870.
"""
heading('E-9.8')
print(q)

xbar = 871
s = 21
n = 50
alpha = 0.05
mu0 = 880
mu1 = 870

pe = xbar
se = s/math.sqrt(n)
z = (pe - mu0)/se
p = stats.norm.cdf(z)
p_value = 1 - p

print(f'mu0: {mu0}')

ci = stats.norm.interval(1-alpha, loc=mu0, scale=se)

print(f'ci: {ci}')
print(f'se: {se}')

print(f'mu1: {mu1}')

beta = stats.norm.cdf(ci[1], loc=mu1, scale=se) - stats.norm.cdf(ci[0], loc=mu1, scale=se)
print(f'beta: {beta}')

power = 1 - beta
print(f'power: {power}')

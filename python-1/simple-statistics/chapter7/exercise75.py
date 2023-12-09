import math
import numpy as np
import pandas as pd

import random

from scipy import stats

from matplotlib import pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Chapter 7: Sampling Distributions')

q = """Random samples of size n were selected from
populations with the means and variances given here.
Find the mean and standard deviation of the sampling
distribution of the sample mean in each case:
a. n = 36, m = 10, s2 = 9
b. n = 100, m = 5, s2 = 4
c. n = 8, m = 120, s2 = 1
"""
heading('Q 7.19')
print(q)

# a
n = 36
m = 10
s2 = 9

m_sample = m
sd_sample = math.sqrt(s2 / n)

print(f'a. m_sample = {m_sample:.4f}')
print(f'   sd_sample = {sd_sample:.4f}')

# b
n = 100
m = 5
s2 = 4

m_sample = m
sd_sample = math.sqrt(s2 / n)

print(f'b. m_sample = {m_sample:.4f}')
print(f'   sd_sample = {sd_sample:.4f}')

# c
n = 8
m = 120
s2 = 1

m_sample = m
sd_sample = math.sqrt(s2 / n)

print(f'c. m_sample = {m_sample:.4f}')
print(f'   sd_sample = {sd_sample:.4f}')


q = """7.25 Suppose a random sample of n = 25 observa-
tions is selected from a population that is normally dis-
tributed with mean equal to 106 and standard deviation
equal to 12.
a. Give the mean and the standard deviation of the
sampling distribution of the sample mean X.
b. Find the probability that X exceeds 110.
c. Find the probability that the sample mean deviates
from the population mean m = 106 by no more
than 4.
"""
heading('Q 7.25')
print(q)

# a
n = 25
m = 106
sd = 12

m_sample = m
sd_sample = sd / math.sqrt(n)

print(f'a. m_sample = {m_sample:.4f}')
print(f'   sd_sample = {sd_sample:.4f}')

f = lambda x: stats.norm.cdf(x, loc=m_sample, scale=sd_sample)

# b
print(f'b. P(X > 110) = {1 - f(110):.4f}')

# c
print(f'c. P({m - 4} < X < {m + 4}) = {f(m + 4) - f(m - 4):.4f}')

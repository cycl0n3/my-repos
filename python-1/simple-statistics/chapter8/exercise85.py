import math
import numpy as np
import pandas as pd

import random

from scipy import stats

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Chapter 8: Estimation')

q = """8.22 Find and interpret a 95% confidence interval for a population mean m for these values:
a. n = 36, m = 13.1, s2 = 3.42
b. n = 64, m = 2.73, s2 = .1047
"""
heading('Q 8.22')
print(q)

# a
n = 36
xbar = 13.1
s = math.sqrt(3.42)

moe = stats.t.ppf(.975, n - 1) * s / math.sqrt(n)

print(f'a.  moe = {moe:.4f}')
print(f'    95% CI = ({xbar - moe:.4f}, {xbar + moe:.4f})')

# b
n = 64
xbar = 2.73
s = math.sqrt(.1047)

moe = stats.t.ppf(.975, n - 1) * s / math.sqrt(n)

print(f'b.  moe = {moe:.4f}')
print(f'    95% CI = ({xbar - moe:.4f}, {xbar + moe:.4f})')


q = """8.23 Find a 90% confidence interval for a population mean m for these values:
a. n = 125, x = .84, s2 = .086
b. n = 50, x = 21.9, s2 = 3.44
c. Interpret the intervals found in parts a and b.
"""
heading('Q 8.23')
print(q)

# a
n = 125
xbar = .84
s = math.sqrt(.086)

moe = stats.t.ppf(.95, n - 1) * s / math.sqrt(n)

print(f'a.  moe = {moe:.4f}')
print(f'    90% CI = ({xbar - moe:.4f}, {xbar + moe:.4f})')

# b
n = 50
xbar = 21.9
s = math.sqrt(3.44)

moe = stats.t.ppf(.95, n - 1) * s / math.sqrt(n)

print(f'b.  moe = {moe:.4f}')
print(f'    90% CI = ({xbar - moe:.4f}, {xbar + moe:.4f})')

# c
print('c. The interval in part a is narrower than the interval in part b because the sample size is larger in part a.')

q = """8.25 A random sample of n = 300 observations from a binomial population produced x = 263 successes.
Find a 90% confidence interval for p and interpret the interval.
"""
heading('Q 8.25')
print(q)

n = 300
x = 263

pbar = x / n
moe = stats.norm.ppf(.95) * math.sqrt(pbar * (1 - pbar) / n)

print(f'moe = {moe:.4f}')
print(f'90% CI = ({pbar - moe:.4f}, {pbar + moe:.4f})')

print('The interval is narrow because the sample size is large.')

q = """8.31 Acid rain, caused by the reaction of certain air pollutants with rainwater, appears to
be a growing problem in the northeastern United States. (Acid rain affects the soil and causes corro-
sion on exposed metal surfaces.) Pure rain falling through clean air registers a pH value of 5.7 (pH is a
measure of acidity: 0 is acid; 14 is alkaline). Suppose water samples from 40 rainfalls are analyzed for pH,
and x� and s are equal to 3.7 and .5, respectively. Find a 99% confidence interval for the mean pH in
rainfall and interpret the interval. What assumption must be made for the confidence interval to be
valid?
"""
heading('Q 8.31')
print(q)

n = 40
xbar = 3.7
s = .5

moe = stats.t.ppf(.995, n - 1) * s / math.sqrt(n)

print(f'moe = {moe:.4f}')
print(f'99% CI = ({xbar - moe:.4f}, {xbar + moe:.4f})')

print('The assumption is that the sample is a random sample from the population of interest.')
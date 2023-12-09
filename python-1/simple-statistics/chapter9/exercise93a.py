import math
import numpy as np
import pandas as pd

import random

from scipy import stats

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Topic 9.3 A Large-Sample Test for the Population Mean')

q = """9.3 Find the appropriate rejection regions for the large-sample test statistic z in these cases:
a. A right-tailed test with alpha = .01
b. A two-tailed test at the 5% significance level
c. A left-tailed test at the 1% significance level
d. A two-tailed test with alpha = .01
"""
heading('9.3')
print(q)

print('a. A right-tailed test with alpha = .01')
print('z > 2.33')

print('b. A two-tailed test at the 5% significance level')
print('z < -1.96 or z > 1.96')

print('c. A left-tailed test at the 1% significance level')
print('z < -2.33')

print('d. A two-tailed test with alpha = .01')
print('z < -2.58 or z > 2.58')

q = """9.4 Find the p-value for the following large-sample z tests:
a. A right-tailed test with observed z = 1.15
b. A two-tailed test with observed z = -2.78
c. A left-tailed test with observed z = -1.81
"""
heading('9.4')
print(q)

# a
z = 1.15
p = 1 - stats.norm.cdf(z)
print(f'a.  p: {p}')

# b
z = -2.78
p = 2 * stats.norm.cdf(z)
print(f'b.  p: {p}')

# c
z = -1.81
p = stats.norm.cdf(z)
print(f'c.  p: {p}')

q = """9.6 A random sample of n = 35 observations from a quantitative population produced a mean xbar = 2.4 and
a standard deviation s = .29. Suppose your research objective is to show that the population mean mu exceeds 2.3. 
a. Give the null and alternative hypotheses for the test.
b. Locate the rejection region for the test using a 5% significance level.
c. Find the standard error of the mean.
d. Before you conduct the test, use your intuition to decide whether the sample mean xbar = 2.4 is likely
or unlikely, assuming that mu = 2.3. Now conduct the test. Do the data provide sufficient evidence to
indicate that mu = 2.3?
"""
heading('9.6')
print(q)

# a
print('a. Give the null and alternative hypotheses for the test.')
print('H0: mu = 2.3')
print('Ha: mu > 2.3')

# b
print('b. Locate the rejection region for the test using a 5% significance level.')
print('z > 1.645')

# c
print('c. Find the standard error of the mean.')
print(f'se = s/sqrt(n) = {0.29/math.sqrt(35)}')

# d
mu0 = 2.3
mu1 = 2.4
se = 0.29/math.sqrt(35)

ci = stats.norm.interval(0.95, loc=mu0, scale=se)
print(f'ci: {ci}')

beta = stats.norm.cdf(ci[1], loc=mu1, scale=se) - stats.norm.cdf(ci[0], loc=mu1, scale=se)
print(f'beta: {beta}')

power = 1 - beta
print(f'power: {power}')

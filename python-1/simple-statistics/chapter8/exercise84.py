import math
import numpy as np
import pandas as pd

import random

from scipy import stats

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Chapter 8: Estimation')

q = """8.3 Calculate the margin of error in estimating a pop-
ulation mean m for these values:
a. n = 30, s^2 = .2
b. n = 30, s^2 = .9
c. n = 30, s^2 = 1.5
"""
heading('Q 8.3')
print(q)

# a
n = 30
s2 = .2

sd = math.sqrt(s2)
moe = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)

print(f'a. moe = {moe:.4f}')

# b
n = 30
s2 = .9

sd = math.sqrt(s2)
moe = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)

print(f'b. moe = {moe:.4f}')

# c
n = 30
s2 = 1.5

sd = math.sqrt(s2)
moe = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)

print(f'c. moe = {moe:.4f}')

q = """8.5 Calculate the margin of error in estimating a pop-ulation mean m for these values:
a. n = 50, s^2 = 4
b. n = 500, s^2 = 4
c. n = 5000, s^2 = 4
"""
heading('Q 8.5')
print(q)

# a
n = 50
s2 = 4

sd = math.sqrt(s2)
moe = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)

print(f'a. moe = {moe:.4f}')

# b
n = 500
s2 = 4

sd = math.sqrt(s2)
moe = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)

print(f'b. moe = {moe:.4f}')

# c
n = 5000
s2 = 4

sd = math.sqrt(s2)
moe = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n)

print(f'c. moe = {moe:.4f}')

q = """8.7 Calculate the margin of error in estimating a binomial proportion for each of the following values
of n. Use p = .5 to calculate the standard error of the estimator.
a. n = 30
b. n = 100
c. n = 400
d. n = 1000
"""
heading('Q 8.7')
print(q)

# a
n = 30
p = .5

se = math.sqrt(p * (1 - p) / n)
moe = stats.norm.ppf(.975) * se

print(f'a. moe = {moe:.4f}')

# b
n = 100
p = .5

se = math.sqrt(p * (1 - p) / n)
moe = stats.norm.ppf(.975) * se

print(f'b. moe = {moe:.4f}')

# c
n = 400
p = .5

se = math.sqrt(p * (1 - p) / n)
moe = stats.norm.ppf(.975) * se

print(f'c. moe = {moe:.4f}')

# d
n = 1000
p = .5

se = math.sqrt(p * (1 - p) / n)
moe = stats.norm.ppf(.975) * se

print(f'd. moe = {moe:.4f}')

q = """8.9 Calculate the margin of error in estimating a binomial proportion p using samples of size n = 100
and the following values for p:
a. p = .1
b. p = .3
c. p = .5
d. p = .7
e. p = .9
f. Which of the values of p produces the largest mar-gin of error?
"""
heading('Q 8.9')
print(q)

# a
n = 100
p = .1

se = math.sqrt(p * (1 - p) / n)
moe = stats.norm.ppf(.975) * se

print(f'a. moe = {moe:.4f}')

# b
n = 100
p = .3

se = math.sqrt(p * (1 - p) / n)
moe = stats.norm.ppf(.975) * se

print(f'b. moe = {moe:.4f}')

# c
n = 100
p = .5

se = math.sqrt(p * (1 - p) / n)
moe = stats.norm.ppf(.975) * se

print(f'c. moe = {moe:.4f}')

# d
n = 100
p = .7

se = math.sqrt(p * (1 - p) / n)
moe = stats.norm.ppf(.975) * se

print(f'd. moe = {moe:.4f}')

# e
n = 100
p = .9

se = math.sqrt(p * (1 - p) / n)
moe = stats.norm.ppf(.975) * se

print(f'e. moe = {moe:.4f}')

# f
print('f. p = .5 produces the largest margin of error')
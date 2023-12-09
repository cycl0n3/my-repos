import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

# Q25
q = """
Consider a binomial experiment with two trials and p = .4.
a. Draw a tree diagram for this experiment (see Figure 5.3).
b. Compute the probability of one success, f(1).
c. Compute f(0).
d. Compute f(2).
e. Compute the probability of at least one success.
f. Compute the expected value, variance, and standard deviation.
"""
heading('Q25')
print(q)

p = .4
q = 1 - p
n = 2

f = lambda x: (math.factorial(n) / (math.factorial(x) * math.factorial(n - x))) * p**x * q**(n - x)

print('b. f(1) =', f(1))
print('c. f(0) =', f(0))
print('d. f(2) =', f(2))

print('e. P(at least one success) =', f(1) + f(2))

E_x = n * p
print('f. E(x) =', E_x)

s2 = n * p * q
print('   s^2 =', s2)

s = np.sqrt(s2)
print('   s =', s)

# Q26
q = """
Consider a binomial experiment with n = 10 and p = .10.
a. Compute f(0).
b. Compute f(2).
c. Compute P(x <= 2).
d. Compute P(x >= 1).
e. Compute E(x).
f. Compute Var(x) and S(x).
"""
heading('Q26')
print(q)

p = .10
q = 1 - p
n = 10

f = lambda x: (math.factorial(n) / (math.factorial(x) * math.factorial(n - x))) * p**x * q**(n - x)

print('a. f(0) =', f(0))
print('b. f(2) =', f(2))

print('c. P(x <= 2) =', f(0) + f(1) + f(2))

print('d. P(x >= 1) =', 1 - f(0))

E_x = n * p
print('e. E(x) =', E_x)

s2 = n * p * q
print('   s^2 =', s2)

s = np.sqrt(s2)
print('   s =', s)

# Q29
q = """In San Francisco, 30% of workers take public transportation daily (USA Today, December
21, 2005).
a. In a sample of 10 workers, what is the probability that exactly three workers take
public transportation daily?
b. In a sample of 10 workers, what is the probability that at least three workers take public
transportation daily?
"""
heading('Q29')
print(q)

p = .30
q = 1 - p
n = 10

f = lambda x: (math.factorial(n) / (math.factorial(x) * math.factorial(n - x))) * p**x * q**(n - x)

print('a. P(x = 3) =', f(3))

print('b. P(x >= 3) =', 1 - f(0) - f(1) - f(2))

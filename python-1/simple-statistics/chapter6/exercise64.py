import math
import numpy as np
import pandas as pd

from scipy import stats

from matplotlib import pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Chapter 6: Continous Probability Distributions')

q = """Consider the following exponential probability density function.
f(x) = (1/8)*e^(-x/8), x >= 0
a. Find P(x <= 6).
b. Find P(x <= 4).
c. Find P(x >= 6).
d. Find P(4 <= x <= 6).
"""
heading('Q32')
print(q)

f = stats.expon(scale=8)

# a
print('a. P(x <= 6) = ', f.cdf(6))

# b
print('b. P(x <= 4) = ', f.cdf(4))

# c
print('c. P(x >= 6) = ', 1 - f.cdf(6))

# d
print('d. P(4 <= x <= 6) = ', f.cdf(6) - f.cdf(4))


q = """The time required to pass through security screening at the airport can be annoying to trav-
elers. The mean wait time during peak periods at Cincinnati/Northern Kentucky Interna-
tional Airport is 12.1 minutes (The Cincinnati Enquirer, February 2, 2006). Assume the
time to pass through security screening follows an exponential distribution.
a. What is the probability it will take less than 10 minutes to pass through security screen-
ing during a peak period?
b. What is the probability it will take more than 20 minutes to pass through security
screening during a peak period?
c. What is the probability it will take between 10 and 20 minutes to pass through secu-
rity screening during a peak period?
d. It is 8:00 A.M. (a peak period) and you just entered the security line. To catch your plane
you must be at the gate within 30 minutes. If it takes 12 minutes from the time you
clear security until you reach your gate, what is the probability you will miss your
flight?
"""
heading('Q34')
print(q)

f = stats.expon(scale=12.1)

# a
print('a. P(x <= 10) = ', f.cdf(10))

# b
print('b. P(x >= 20) = ', 1 - f.cdf(20))

# c
print('c. P(10 <= x <= 20) = ', f.cdf(20) - f.cdf(10))

# d
print('d. P(x >= 18) = ', 1 - f.cdf(18))


q = """Do interruptions while you are working reduce your productivity? According to a Univer-
sity of California–Irvine study, businesspeople are interrupted at the rate of approximately
51⁄2 times per hour (Fortune, March 20, 2006). Suppose the number of interruptions fol-
lows a Poisson probability distribution.
a. Show the probability distribution for the time between interruptions.
b. What is the probability a businessperson will have no interruptions during a 15-minute
period?
c. What is the probability that the next interruption will occur within 10 minutes for a
particular businessperson?
"""
heading('Q38')
print(q)

# a
# mu = 5.5
# f = stats.poisson(mu)

# x = np.arange(0, 20)
# y = f.pmf(x)

# plt.bar(x, y)
# plt.show()

# b
mu = 5.5 / 4
f = stats.poisson(mu)
print('b. P(x = 0) = ', f.pmf(0))

# c
mu = 60 / 5.5
f = stats.expon(scale=mu)
print('c. P(x <= 10) = ', f.cdf(10))

import math
import numpy as np
import pandas as pd

from scipy import stats

from matplotlib import pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Chapter 6: Continous Probability Distributions')


q = """A binomial probability distribution has p = .20 and n = 100.
a. What are the mean and standard deviation?
b. Is this situation one in which binomial probabilities can be approximated by the nor-
mal probability distribution? Explain.
c. What is the probability of exactly 24 successes?
d. What is the probability of 18 to 22 successes?
e. What is the probability of 15 or fewer successes?
"""
heading('Q26')
print(q)

p = 0.2
n = 100

mu = n * p
s = math.sqrt(n * p * (1 - p))

# a
f = stats.binom(n, p)

print('a. mu = ', mu)
print('   s = ', s)

# b
print('b. Yes, because n*p = 20 > 5 and n*(1-p) = 80 > 5')

# c
print('c. P(x = 24) = ', f.cdf(24.5) - f.cdf(23.5))

# d
print('d. P(18 <= x <= 22) = ', f.cdf(22.5) - f.cdf(17.5))

# e
print('e. P(x <= 15) = ', f.cdf(15.5))


q = """An Internal Revenue Oversight Board survey found that 82% of taxpayers said that it was
very important for the Internal Revenue Service (IRS) to ensure that high-income tax pay-
ers do not cheat on their tax returns (The Wall Street Journal, February 11, 2009).
a. For a sample of eight taxpayers, what is the probability that at least six taxpayers say
that it is very important to ensure that high-income tax payers do not cheat on their tax
returns? Use the binomial distribution probability function shown in Section 5.4 to an-
swer this question.
b. For a sample of 80 taxpayers, what is the probability that at least 60 taxpayers say that
it is very important to ensure that high-income tax payers do not cheat on their tax
returns? Use the normal approximation of the binomial distribution to answer this
question.
c. As the number of trails in a binomial distribution application becomes large, what is
the advantage of using the normal approximation of the binomial distribution to com-
pute probabilities?
d. When the number of trials for a binominal distribution application becomes large,
would developers of statistical software packages prefer to use the binomial distribu-
tion probability function shown in Section 5.4 or the normal approximation of the
binomial distribution shown in Section 6.3? Explain.
"""
heading('Q29')
print(q)

p = 0.82
n = 8

f = stats.binom(n, p)

# a
print('a. P(x >= 6) = ', 1 - f.cdf(5))

# b
p = 0.82
n = 80
f = stats.binom(n, p)

print('b. P(x >= 60) = ', 1 - f.cdf(59))

# c
print('c. The normal approximation is easier to compute.')

# d
print('d. The normal approximation is easier to compute.')
import math
import numpy as np
import pandas as pd

import random

from scipy import stats

from matplotlib import pyplot as plt

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Chapter 7: Sampling Distributions')

q = """The duration of Alzheimer’s disease from the onset of symptoms until death ranges
from 3 to 20 years; the average is 8 years with a standard deviation of 4 years. The
administrator of a large medical center randomly selects the medical records of 30 de-
ceased Alzheimer’s patients from the medical center’s database, and records the av-
erage duration. Find the approximate probabilities for these events:
1. The average duration is less than 7 years.
2. The average duration exceeds 7 years.
3. The average duration lies within 1 year of the population mean m = 8.
"""
heading('E 7.4')
print(q)

m = 8
sd = 4
n = 30

m_sample = m
sd_sample = sd / math.sqrt(n)

f = lambda x: stats.norm.cdf(x, loc=m_sample, scale=sd_sample)

print(f'Sample mean = {m_sample:.4f}')
print(f'Sample standard deviation = {sd_sample:.4f}')

# 1
print(f'1. P(X < 7) = {f(7):.4f}')

# 2
print(f'2. P(X > 7) = {1 - f(7):.4f}')

# 3
print(f'3. P({m - 1} < X < {m + 1}) = {f(m + 1) - f(m - 1):.4f}')


q = """To avoid difficulties with the Federal Trade Commission or state and local consumer
protection agencies, a beverage bottler must make reasonably certain that 12-ounce
bottles actually contain 12 ounces of beverage. To determine whether a bottling ma-
chine is working satisfactorily, one bottler randomly samples 10 bottles per hour and
measures the amount of beverage in each bottle. The mean x� of the 10 fill measure-
ments is used to decide whether to readjust the amount of beverage delivered per bot-
tle by the filling machine. If records show that the amount of fill per bottle is normally
distributed, with a standard deviation of .2 ounce, and if the bottling machine is set
to produce a mean fill per bottle of 12.1 ounces, what is the approximate probability
that the sample mean x� of the 10 test bottles is less than 12 ounces?
"""
heading('E 7.5')
print(q)

m = 12.1
sd = 0.2

m_sample = m
sd_sample = sd / math.sqrt(10)

f = lambda x: stats.norm.cdf(x, loc=m_sample, scale=sd_sample)

print(f'Sample mean = {m_sample:.4f}')
print(f'Sample standard deviation = {sd_sample:.4f}')

print(f'P(X < 12) = {f(12):.4f}')

import math
import numpy as np
import pandas as pd

import random

from scipy import stats

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Chapter 8: Estimation')

q = """8.13 The San Andreas Fault Geologists are inter-ested in shifts and movements of the earth’s surface
indicated by fractures (cracks) in the earth’s crust. One of the most famous large fractures is the San Andreas
fault in California. A geologist attempting to study the movement of the relative shifts in the earth’s crust at a
particular location found many fractures in the local rock structure. In an attempt to determine the mean
angle of the breaks, she sampled n = 50 fractures and found the sample mean and standard deviation to be
39.8° and 17.2°, respectively. Estimate the mean angu-lar direction of the fractures and find the margin of
error for your estimate.
"""
heading('Q 8.13')
print(q)

n = 50
xbar = 39.8
s = 17.2

moe = stats.t.ppf(.975, n - 1) * s / math.sqrt(n)

print(f'moe = {moe:.4f}')

q = """8.14 Biomass Estimates of the earth’s biomass, the total amount of vegetation held by the earth’s forests,
are important in determining the amount of unab-sorbed carbon dioxide that is expected to remain in
the earth’s atmosphere.2 Suppose a sample of 75 one-square-meter plots, randomly chosen in North
America’s boreal (northern) forests, produced a mean biomass of 4.2 kilograms per square meter (kg/m2),
with a standard deviation of 1.5 kg/m2. Estimate the average biomass for the boreal forests of North America
and find the margin of error for your estimate.
"""
heading('Q 8.14')
print(q)

n = 75
xbar = 4.2
s = 1.5

moe = stats.t.ppf(.975, n - 1) * s / math.sqrt(n)

print(f'moe = {moe:.4f}')
print(f'95% CI = ({xbar - moe:.4f}, {xbar + moe:.4f})')
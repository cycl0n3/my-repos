import math
import numpy as np
import pandas as pd

import random

from scipy import stats

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Chapter 8: Estimation')

q = """E 8.9 The wearing qualities of two types of automobile tires were compared by road-test-
ing samples of n1 = n2 = 100 tires for each type. The number of miles until wearout
was defined as a specific amount of tire wear. The test results are given in Table 8.4.
Estimate (m1 - m2), the difference in mean miles to wearout, using a 99% confidence
interval. Is there a difference in the average wearing quality for the two types of tires?
Tire 1                              Tire 2
x1 = 26,400 miles                  x2 = 25,100 miles
s1^2 = 1,440,000                    s2^2 = 1,960,000
"""
heading('8.9')
print(q)

x1 = 26400
x2 = 25100
s1 = math.sqrt(1440000)
s2 = math.sqrt(1960000)
n1 = 100
n2 = 100
alpha = 0.01

pe = x1 - x2
se = math.sqrt((s1**2/n1) + (s2**2/n2))
z = stats.norm.ppf(1 - alpha/2)
ci = (pe - z*se, pe + z*se)

print(f'Confidence interval: {ci}')
print(f'Conclusion: There is a difference in the average wearing quality for the two types of tires.')

q = """E 8.10 The scientist wondered whether there was a difference in the average
daily intakes of dairy products between men and women. He took a sample of
n = 50 adult women and recorded their daily intakes of dairy products in grams
per day. He did the same for adult men. A summary of his sample results is listed in
Table. Construct a 95% confidence interval for the difference in the average daily
intakes of dairy products for men and women. Can you conclude that there is a dif-
ference in the average daily intakes for men and women?
                            Men                             Women
Sample Size                 50                              50
Sample Mean                 756                             762
Sample Standard Deviation   35                              30
"""
heading('8.10')
print(q)

x1 = 756
x2 = 762
s1 = 35
s2 = 30
n1 = 50
n2 = 50
alpha = 0.05

pe = x1 - x2
se = math.sqrt((s1**2/n1) + (s2**2/n2))
z = stats.norm.ppf(1 - alpha/2)
ci = (pe - z*se, pe + z*se)

print(f'Confidence interval: {ci}')

q = """8.41 Selenium A small amount of the trace element selenium, 50–200 micrograms (mg) per day, is consid-
ered essential to good health. Suppose that random samples of n1 = n2 = 30 adults were selected from
two regions of the United States and that a day’s intake of selenium, from both liquids and solids, was
recorded for each person. The mean and standard devi-ation of the selenium daily intakes for the 30 adults
from region 1 were x1 = 167.1 and s1 = 24.3 mg, respectively. The corresponding statistics for the
30 adults from region 2 were x2 = 140.9 and s2 = 17.6. Find a 95% confidence interval for the difference
in the mean selenium intakes for the two regions. Interpret this interval.
"""
heading('8.41')
print(q)

x1 = 167.1
x2 = 140.9
s1 = 24.3
s2 = 17.6
n1 = 30
n2 = 30
alpha = 0.05

pe = x1 - x2
se = math.sqrt((s1**2/n1) + (s2**2/n2))
z = stats.norm.ppf(1 - alpha/2)
ci = (pe - z*se, pe + z*se)

print(f'Confidence interval: {ci}')
print(f'Conclusion: The mean selenium intake for region 1 is higher than that of region 2.')
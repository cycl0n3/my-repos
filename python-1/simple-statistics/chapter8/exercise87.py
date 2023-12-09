import math
import numpy as np
import pandas as pd

import random

from scipy import stats

def heading(question_number, p='=', l=50):
    print(f'{p*l}Question {question_number}{p*l}')

print('Topic 8.7: Estimating the difference between two binomial proportions')

q = """E 8.11 A bond proposal for school construction will be submitted to the voters at the next
municipal election. A major portion of the money derived from this bond issue will
be used to build schools in a rapidly developing section of the city, and the remain-
der will be used to renovate and update school buildings in the rest of the city. To
assess the viability of the bond proposal, a random sample of n1 = 50 residents in the
developing section and n2 = 100 residents from the other parts of the city were asked
whether they plan to vote for the proposal. The results are tabulated in Table

                                Developing Section                   Rest of the City
Sample Size                     50                                   100
Number Favoring Proposal        38                                   65
Proportion Favoring Proposal    .76                                  .65

1. Estimate the difference in the true proportions favoring the bond proposal with
a 99% confidence interval.
2. If both samples were pooled into one sample of size n = 150, with 103 in
favor of the proposal, provide a point estimate of the proportion of city resi-
dents who will vote for the bond proposal. What is the margin of error?
"""
heading('E 8.11')
print(q)

n1 = 50
n2 = 100
p1 = 38/50
p2 = 65/100
alpha = 0.01

# 1
pe = p1 - p2
se = math.sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
z = stats.norm.ppf(1 - alpha/2)
ci = (pe - z*se, pe + z*se)

print(f'1.  Confidence interval: {ci}')

# 2
n = 150
p = 103/150
se = math.sqrt((p*(1-p)/n))
z = stats.norm.ppf(1 - alpha/2)
ci = (p - z*se, p + z*se)

print(f'2.  Confidence interval: {ci}')

q = """8.51 Independent random samples of n1 = 800 and n2 = 640 observations were selected from binomial
populations 1 and 2, and x1 = 337 and x2 = 374 suc-cesses were observed.
a. Find a 90% confidence interval for the difference (p1 - p2) 
in the two population proportions. Inter- pret the interval.
b. What assumptions must you make for the confi- dence interval to be valid? Are these assumptions met?
"""
heading('Q 8.51')
print(q)

n1 = 800
n2 = 640
x1 = 337
x2 = 374
alpha = 0.1

# a
p1 = x1/n1
p2 = x2/n2
pe = p1 - p2
se = math.sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
z = stats.norm.ppf(1 - alpha/2)
ci = (pe - z*se, pe + z*se)

print(f'a.  Confidence interval: {ci}')

q = """8.61 Excedrin or Tylenol? In a study to compare the effects of two pain relievers it was found that of
n1 = 200 randomly selectd individuals instructed to use the first pain reliever, 93% indicated that it
relieved their pain. Of n2 = 450 randomly selected individuals instructed to use the second pain reliever,
96% indicated that it relieved their pain.
a. Find a 99% confidence interval for the difference in the proportions experiencing relief from pain for
these two pain relievers.
b. Based on the confidence interval in part a, is there sufficient evidence to indicate a difference in the
proportions experiencing relief for the two pain relievers? Explain.
"""
heading('Q 8.61')
print(q)

n1 = 200
n2 = 450
p1 = 0.93
p2 = 0.96
alpha = 0.01

# a
pe = p1 - p2
se = math.sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
z = stats.norm.ppf(1 - alpha/2)
ci = (pe - z*se, pe + z*se)

print(f'a.  Confidence interval: {ci}')

# b
print(f'b.  No, the confidence interval contains 0, so there is no difference in the proportions.')

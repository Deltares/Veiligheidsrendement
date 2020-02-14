import cProfile
import pstats
import io
from pathlib import Path

path = Path(r'd:\wouterjanklerk\My Documents\00_PhDgeneral\01_Scripts')
s = io.StringIO()
ps = pstats.Stats(str(path.joinpath('SAFE','evaluate_old')),stream=s).sort_stats('cumtime')
ps.print_stats()

with open('testold.txt', 'w+') as f:
    f.write(s.getvalue())
import ProbabilisticFunctions

from scipy.stats import norm
import numpy as np
import cProfile


import math

#
# ###################################################################################################
#
# def erf(x):
#     ''' John D. Cook's implementation.http://www.johndcook.com
#     >> Formula 7.1.26 given in Abramowitz and Stegun.
#     >> Formula appears as 1 – (a1t1 + a2t2 + a3t3 + a4t4 + a5t5)exp(-x2)
#     >> A little wisdom in Horner's Method of coding polynomials:
#         1) We could evaluate a polynomial of the form a + bx + cx^2 + dx^3 by coding as a + b*x + c*x*x + d*x*x*x.
#         2) But we can save computational power by coding it as ((d*x + c)*x + b)*x + a.
#         3) The formula below was coded this way bringing down the complexity of this algorithm from O(n2) to O(n).'''
#
#     # constants
#     a1 = 0.254829592
#     a2 = -0.284496736
#     a3 = 1.421413741
#     a4 = -1.453152027
#     a5 = 1.061405429
#     p = 0.3275911
#
#     # Save the sign of x
#     # sign = 1
#     # if x < 0:
#     #     sign = -1
#     sign = np.where(x < 0, -1, 1)
#     x = abs(x)
#
#     # Formula 7.1.26 given in Abramowitz and Stegun.
#     t = 1.0 / (1.0 + p * x)
#     y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
#     # y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
#
#     return sign * y
#
#
# ####################################################################################################
#
# def phi(x):
#     '''Cumulative gives a probability that a statistic
#     is less than Z. This equates to the area of the
#     distribution below Z.
#     e.g:  Pr(Z = 0.69) = 0.7549. This value is usually
#     given in Z tables.'''
#
#     return 0.5 * (1.0 + erf(x / math.sqrt(2)))
#
#
# #####################################################################################################
#
# def phi_compcum(x):
#     ''' Complementary cumulative gives a probability
#     that a statistic is greater than Z. This equates to
#     the area of the distribution above Z.
#     e.g: Pr(Z  =  0.69) = 1 - 0.7549 = 0.2451'''
#
#     return abs(phi(x) - 1)
#
#
# #####################################################################################################
#
# def phi_cumformu(x):
#     '''Cumulative from mean gives a probability
#     that a statistic is between 0 (mean) and Z.
#     e.g: Pr(0 = Z = 0.69) = 0.2549'''
#
#     return phi_compcum(0) - phi_compcum(x)
#
#
# def formula(t):
#     # constants
#     c0 = 2.515517
#     c1 = 0.802853
#     c2 = 0.010328
#     d0 = 1.432788
#     d1 = 0.189269
#     d2 = 0.001308
#
#     # Formula
#     p = t - ((c2 * t + c1) * t + c0) / (((d2 * t + d1) * t + d0) * t + 1.0)
#     return p
#
#
# def q(p):
#     x1 = -formula(np.sqrt(-2.0 * np.log(p)))
#     x2 = formula(np.sqrt(-2.0 * np.log(1 - p)))
#     return np.where(p<0.5,x1,x2)
#     #
#     # if (p < 0.5):
#     #     # F^-1(p) = - G^-1(p)
#     #     return -formula(math.sqrt(-2.0 * math.log(p)))
#     #
#     # else:
#     #     # F^-1(p) = G^-1(1-p)
#     #     return formula(math.sqrt(-2.0 * math.log(1 - p)))
#
#     return q
# def run_q(p):
#     x = ProbabilisticFunctions.pf_to_beta(p)
#     # x = []
#     # for i in p:
#     #     x.append(q(i))
#     return x
#
# def run_normal_ppf(p):
#     x = -norm.ppf(p)
#     # x = []
#     # for i in p:
#     #     x.append(norm.ppf(p))
#     return x
# def run_phi(x):
#     p = ProbabilisticFunctions.beta_to_pf(x)
#     # p = []
#     # for i in x:
#     #     p.append(phi(i))
#     return p
# def run_normal_cdf(x):
#     p = norm.cdf(-x)
#     # p = []
#     # for i in x:
#     #     p.append(norm.cdf(i))
#     return p
# # norm.cdf
# # cProfile.run('run_normal_cdf(x)')
# # cProfile.run('run_phi(x)')
#
# x = np.linspace(0,8,10000).astype(np.float32)
# p1 = run_normal_cdf(x)
# p2 = run_phi(x)
# import matplotlib.pyplot as plt
# plt.plot(x,p1,'xr')
# plt.plot(x,p2,'ob')
# plt.plot(x,np.subtract(p1,p2),'k*')
# plt.yscale('log')
# plt.show()
#
# # p = np.logspace(-1,-16,10000)
# # cProfile.run('run_normal_ppf(p)')
# # cProfile.run('run_q(p)')
# # #     # np.random.rand(10000)
# # x1 = run_normal_ppf(p)
# # x2 = run_q(p)
# # import matplotlib.pyplot as plt
# # plt.plot(x1,'xr')
# # plt.plot(x2,'ob')
# # plt.plot(np.divide(x1,x2),'k*')
# # #
# # plt.show()
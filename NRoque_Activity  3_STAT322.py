# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:27:33 2024

@author: NRoque
"""

def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    
    not_a = 1 - p_a
    
    p_b = p_b_given_a * p_a + p_b_given_not_a * not_a
    
    p_a_given_b = (p_b_given_a * p_a) / p_b
    
    return p_a_given_b


p_a = 0.0002

p_b_given_a = 0.85

p_b_given_not_a = 0.05

result = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)

print('P(A|B) = %.3f%%' % (result * 100))



import numpy as np
import matplotlib.pyplot as plt

prior_probs = np.array([[0.33,0.3], [0.2,0.17]])

plt.imshow(prior_probs, cmap= 'gray')
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.annotate(prior_probs[i,j], (j,i), color="red" , fontsize=20, fontweight='bold', ha='center', va='center')

plt.title('Prior Probabilities', fontsize=20)
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:16:29 2024

@author: Home
"""

import scipy.stats as sts 
import numpy as np 
import matplotlib.pyplot as plt 
mu = np.linspace (1.65, 1.8, num = 50) 
test = np.linspace(0, 2) 
uniform_dist = sts.uniform.pdf(mu) + 1 #sneaky advanced note: I'm using the uniform distribution for clarity, 
#but we can also make the beta distribution look completely flat by tweaking alpha and beta! 
uniform_dist = uniform_dist/uniform_dist.sum() #Normalizing the distribution to make the probability densities sum into 1 
beta_dist = sts.beta.pdf(mu, 2, 5, loc = 1.65, scale = 0.2) 
beta_dist = beta_dist/beta_dist.sum() 
plt.plot(mu, beta_dist, label = 'Beta Dist') 
plt.plot(mu, uniform_dist, label = 'Uniform Dist') 
plt.xlabel("Value of $\mu$ in meters") 
plt.ylabel("Probability Density") 
plt.legend()

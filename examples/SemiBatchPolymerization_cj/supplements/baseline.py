#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:34:20 2018

@author: flemmingholtorf
"""

from __future__ import print_function
from pyomo.environ import *
import sys
import itertools, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import pickle
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *
from scipy.stats import chi2
from copy import deepcopy
from main.dync.MHEGen_adjusted import MheGen
from main.mods.final_pwa.mod_class_cj_pwa import * #_robust_optimal_control
from main.noise_characteristics_cj import * 


mod = SemiBatchPolymerization(24,3)
mod.initialize_element_by_element()
mod.create_bounds()
mod.clear_aux_bounds()
ip = SolverFactory('ipopt')
ip.solve(mod, tee=True)
nom_Ap = mod.A['p'].value
nom_Ai = mod.A['i'].value
nom_kA = mod.kA.value
grid = [-0.2,-0.1,0.0,0.1,0.2]
kA = np.array(grid)#np.linspace(-0.2,0.2,num=4)
Ap = np.array(grid)#np.linspace(-0.2,0.2,num=4)
Ai = np.array(grid)#np.linspace(-0.2,0.2,num=4)
Ap, Ai, kA = np.meshgrid(kA, Ai, Ap)
mod.tf.setub(None)
i = 0
scenarios = {}
n = len(grid)
for j in range(n):
    for k in range(n):
        for l in range(n):
            scenarios[i] = {('A',('p',)):Ap[j][k][l],('A',('i',)):Ai[j][k][l],('kA',()):kA[j][k][l]}
            i += 1
sample_size = n**3
tf_lb = {}
for i in range(sample_size):
    mod.A['p'] = nom_Ap * (1+scenarios[i][('A',('p',))])
    mod.A['i'] = nom_Ai * (1+scenarios[i][('A',('i',))])
    mod.kA = nom_kA *  (1+scenarios[i][('kA',())])
    ip.solve(mod, tee=True)
    tf_lb[i] = mod.tf.value

tf_lb = {key:tf_lb[key]*24.0 for key in tf_lb}
min_tf = min([tf_lb[key] for key in tf_lb]) 
max_tf = max([tf_lb[key] for key in tf_lb])
avg_tf = sum(tf_lb[key] for key in tf_lb)/len(tf_lb)

path = 'temp/'

f = open(path + 'lower_bound.pckl','wb')
pickle.dump(tf_lb, f)
f.close()

f = open(path + 'scenarios.pckl','wb')
pickle.dump(scenarios, f)
f.close()
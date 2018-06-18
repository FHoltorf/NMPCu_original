#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:15:13 2018

@author: flemmingholtorf
"""
import pickle, time
import numpy as np
from main.examples.SemiBatchPolymerization.MC_sampling.controllers.msMHE_stgen import *
#from main.examples.SemiBatchPolymerization.MC_sampling.controllers.msMHE import *
#from main.examples.SemiBatchPolymerization.MC_sampling.controllers.MHE_linapprox import *
#from main.examples.SemiBatchPolymerization.MC_sampling.controllers.MHE import *
from copy import deepcopy

grid = [-0.2,-0.1,0.0,0.1,0.2]
kA = np.array(grid)#np.linspace(-0.2,0.2,num=4)
Ap = np.array(grid)#np.linspace(-0.2,0.2,num=4)
Ai = np.array(grid)#np.linspace(-0.2,0.2,num=4)
Ap, Ai, kA = np.meshgrid(kA, Ap, Ai)
i = 0
scenarios = {}
n = len(grid)
for j in range(n):
    for k in range(n):
        for l in range(n):
            scenarios[i] = {('A',('p',)):Ap[j][k][l],('A',('i',)):Ai[j][k][l],('kA',()):kA[j][k][l]}
            i += 1
sample_size = n**3

path = 'results/'
timestamp = int(time.time())

tf = {}
endpoint_constraints = {}
path_constraints = {}
runtime = {} # runtime in seconds
uncertainty_realization = {}
nmpc_trajectory = {}
monitor = {}
CPU_t = {}
for i in range(sample_size):
    print('#'*20 + '\n' + ' '*4 + 'iteration:' + str(i) + '\n' + '#'*20)
    disturbance_src = {'disturbance_src':'parameter_scenario','scenario':scenarios[i]}
    args['disturbance_src'] = disturbance_src
    c = deepcopy(cl)
    # run closed-loop simulation:
    performance, iters = c.run(**args)
    if iters == 24:            
        tf[i] = c.nmpc_trajectory[iters,'tf']
        endpoint_constraints[i] = c.plant_simulation_model.check_feasibility(display=True)
        path_constraints[i] = deepcopy(c.pc_trajectory)
        CPU_t[i] = performance
        nmpc_trajectory[i] = deepcopy(c.nmpc_trajectory)
        monitor[i] = deepcopy(c.monitor)
    else:
        tf[i]=endpoint_constraints[i]=path_constraints[i]=CPU_t[i]=nmpc_trajectory[i]=monitor[i]='crashed'
    
""" store raw data """
path += kind+parest+str(timestamp)+'.pckl'
pickle.dump({'tf':tf,'epc':endpoint_constraints,'pc':path_constraints,'t_CPU':CPU_t,
             'nmpc_trajectory':nmpc_trajectory,'monitor':monitor,'scenarios':scenarios},
            open(path,'wb'))
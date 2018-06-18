#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:07:29 2018

@author: flemmingholtorf
"""
import pickle
import numpy as np

constraint_name = []

paths = ['crashedruns/bestsofar/beginning/','crashedruns/bestsofar/rest/']
epc_patches = {}
final_times_patches = {}
path_constraints_patches = {}
uncertainty_realization_patches = {}
runtime_patches = {}
CPU_t_patches = {}
scenarios_patches = {}
k = 0
for path in paths:
    # load 
    f = open(path + 'epc.pckl','rb')
    epc_patches[k] = pickle.load(f)
    f.close()
    
    f = open(path + 'final_times.pckl','rb')
    final_times_patches[k] = pickle.load(f)
    f.close()
    
    f = open(path + 'path_constraints.pckl','rb')
    path_constraints_patches[k] = pickle.load(f)
    f.close()
    
    f = open(path + 'uncertainty_realization.pckl','rb')
    uncertainty_realization_patches[k] = pickle.load(f)
    f.close()
    
    f = open(path + 'runtime.pckl','rb')
    runtime_patches[k] = pickle.load(f)
    f.close()
    
    f = open(path + 'CPU_t.pckl','rb')
    CPU_t_patches[k] = pickle.load(f)
    f.close()
    
    f = open(path + 'scenarios.pckl','rb')
    scenarios_patches[k] = pickle.load(f)
    f.close()
    k += 1

epc = {}
final_times = {}
path_constraints = {}
uncertainty_realization = {}
runtime = {}
CPU_t = {}
scenarios = {}
endpoint_constraints = {key:value for i in range(k) for key,value in epc_patches[i].items()}
path_constraints = {key:value for i in range(k) for key,value in path_constraints_patches[i].items()}
uncertainty_realization = {key:value for i in range(k) for key,value in uncertainty_realization_patches[i].items()}
runtime = {key:value for i in range(k) for key,value in runtime_patches[i].items()}
CPU_t = {key:value for i in range(k) for key,value in CPU_t_patches[i].items()}
scenarios = {key:value for i in range(k) for key,value in scenarios_patches[i].items()}
tf = {key:value for i in range(k) for key,value in final_times_patches[i].items()}
# save patched up stuff

path = 'temp/'
f = open(path + 'epc.pckl','wb')
pickle.dump(endpoint_constraints, f)
f.close()

f = open(path + 'final_times.pckl','wb')
pickle.dump(tf,f)
f.close()

f = open(path + 'path_constraints.pckl','wb')
pickle.dump(path_constraints,f)
f.close()

f = open(path + 'uncertainty_realization.pckl','wb')
pickle.dump(uncertainty_realization,f)
f.close()

f = open(path + 'runtime.pckl','wb')
pickle.dump(runtime,f)
f.close()

f = open(path + 'CPU_t.pckl','wb')
pickle.dump(CPU_t,f)
f.close()

f = open(path + 'scenarios.pckl','wb')
pickle.dump(scenarios,f)
f.close()
# path constraints
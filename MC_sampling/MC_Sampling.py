#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:39:42 2017

@author: flemmingholtorf
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

# model with cooling jacket 
#from main.MC_sampling.cj.run_MHE_NMPC_cj_pwa import *
#from main.MC_sampling.cj.run_MHE_NMPC_cj_pwa_bo import *
#from main.MC_sampling.cj.run_MHE_NMPC_cj_pwa_SBBM import *
#from main.MC_sampling.cj.run_MHE_NMPC_cj_pwa_multistage import *
from main.MC_sampling.cj.run_MHE_NMPC_cj_pwa_multistage_stgen import *
#from main.MC_sampling.cj.run_MHE_asNMPC_cj_pwa_multistage_stgen import *
#from main.MC_sampling.test.runtest import *
#from main.MC_sampling.test.run_MHE_NMPC_stgen_onestage import *

#################################################################
#################################################################
#################################################################
###   CAN NOT RUN EVERYTHING THAT USES dx_dp SIMULTANEOUSLY   ###
#################################################################
#################################################################
#################################################################
# parameter realizations use grid search for worst-case:
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
            
            
#scenarios_wc = {0: {('A', ('i',)): -0.2, ('A', ('p',)): -0.2, ('kA', ()): -0.2},
# 1: {('A', ('i',)): -0.2, ('A', ('p',)): -0.2, ('kA', ()): -0.1},
# 2: {('A', ('i',)): -0.2, ('A', ('p',)): -0.2, ('kA', ()): 0.0},
# 3: {('A', ('i',)): -0.2, ('A', ('p',)): -0.2, ('kA', ()): 0.1},
# 4: {('A', ('i',)): -0.2, ('A', ('p',)): -0.2, ('kA', ()): 0.2},
# 5: {('A', ('i',)): -0.1, ('A', ('p',)): -0.2, ('kA', ()): -0.2},
# 6: {('A', ('i',)): -0.1, ('A', ('p',)): -0.2, ('kA', ()): -0.1},
# 7: {('A', ('i',)): -0.1, ('A', ('p',)): -0.2, ('kA', ()): 0.0},
# 8: {('A', ('i',)): -0.1, ('A', ('p',)): -0.2, ('kA', ()): 0.1},
# 9: {('A', ('i',)): -0.1, ('A', ('p',)): -0.2, ('kA', ()): 0.2},
# 10: {('A', ('i',)): 0.0, ('A', ('p',)): -0.2, ('kA', ()): -0.2},
# 11: {('A', ('i',)): 0.0, ('A', ('p',)): -0.2, ('kA', ()): -0.1},
# 12: {('A', ('i',)): 0.0, ('A', ('p',)): -0.2, ('kA', ()): 0.0},
# 13: {('A', ('i',)): 0.0, ('A', ('p',)): -0.2, ('kA', ()): 0.1},
# 14: {('A', ('i',)): 0.0, ('A', ('p',)): -0.2, ('kA', ()): 0.2},
# 15: {('A', ('i',)): 0.1, ('A', ('p',)): -0.2, ('kA', ()): -0.2},
# 16: {('A', ('i',)): 0.1, ('A', ('p',)): -0.2, ('kA', ()): -0.1},
# 17: {('A', ('i',)): 0.1, ('A', ('p',)): -0.2, ('kA', ()): 0.0},
# 18: {('A', ('i',)): 0.1, ('A', ('p',)): -0.2, ('kA', ()): 0.1},
# 19: {('A', ('i',)): 0.1, ('A', ('p',)): -0.2, ('kA', ()): 0.2},
# 20: {('A', ('i',)): 0.2, ('A', ('p',)): -0.2, ('kA', ()): -0.2},
# 21: {('A', ('i',)): 0.2, ('A', ('p',)): -0.2, ('kA', ()): -0.1},
# 22: {('A', ('i',)): 0.2, ('A', ('p',)): -0.2, ('kA', ()): 0.0},
# 23: {('A', ('i',)): 0.2, ('A', ('p',)): -0.2, ('kA', ()): 0.1},
# 24: {('A', ('i',)): 0.2, ('A', ('p',)): -0.2, ('kA', ()): 0.2}}            
#scenarios = scenarios_wc
# inputs
sample_size = n**3
# specifiy directory where to save the resulting files
path = 'temp/' 
# colors
color = ['green','red','blue']
tf = {}
endpoint_constraints = {}
path_constraints = {}
runtime = {} # runtime in seconds
uncertainty_realization = {}
CPU_t = {}
# run sample_size batches and save the endpoint constraint violation
iters = 0
runs = []
for i in range(sample_size):
    print('#'*20)
    print('#'*20)
    print('#'*20)
    print('#'*20)
    print(' '*5 + 'iter: ' + str(i))
    print('#'*20)
    print('#'*20)
    print('#'*20)
    print('#'*20)
    try:
        t0 = time.time()
        tf[i],endpoint_constraints[i],path_constraints[i],uncertainty_realization[i],CPU_t[i] = run(scenario=scenarios[i])
        runtime[i] = time.time() - t0
    except ValueError:
        tf[i],endpoint_constraints[i],path_constraints[i],uncertainty_realization[i],CPU_t[i] = 'error', {'epc_PO_ptg': 'error', \
                                   'epc_mw': 'error', \
                                   'epc_unsat': 'error'}, 'error', 'error','error'
        runtime[i]  = 0.0 
    
    feasible = True
    for constraint in endpoint_constraints[i]:
        if endpoint_constraints[i][constraint] == 'error':
            feasible = 'crashed'
        elif endpoint_constraints[i][constraint] < 0:
            feasible = False
            break
    endpoint_constraints[i]['feasible'] = feasible
    iters += 1
    runs.append(i)
#    if i == 4:
#       letsgo = input('letsgo')
   
constraint_name = []

for constraint in endpoint_constraints[runs[0]]:
    if constraint == 'feasible':
        continue
    constraint_name.append(constraint)

unit = {'epc_PO_ptg' : ' [PPM]', 'epc_unsat' : ' [mol/g PO]', 'epc_mw' : ' [g/mol]'}

# enpoint constraints 
for k in range(3):
    color[k]
    x = [endpoint_constraints[i][constraint_name[k]] for i in runs if endpoint_constraints[i][constraint_name[k]] != 'error']
    plt.figure(k)
    plt.hist(x,int(np.ceil(iters**0.5)), normed=None, facecolor=color[k], edgecolor='black', alpha=1.0) 
    plt.xlabel(constraint_name[k] + unit[constraint_name[k]])
    plt.ylabel('relative frequency')
    plt.figure(k).savefig(path + constraint_name[k] +'.pdf')
fes = 0
infes = 0

for i in runs:
    # problem is feasible
    if endpoint_constraints[i]['feasible'] == True:
        fes += 1
    elif endpoint_constraints[i]['feasible'] == False:
        infes += 1
sizes = [fes, infes, iters-fes-infes]

plt.figure(3)
plt.axis('equal')
explode = (0.0, 0.1, 0.0) 
wedges= plt.pie(sizes,explode,labels=['feasible','infeasible','crashed'], autopct='%1.1f%%',shadow=True)
for w in wedges[0]:
    w.set_edgecolor('black')
plt.figure(3).savefig(path + 'feas.pdf')
# save results to file using pickle

# compute final time histogram
plt.figure(4)
x = [tf[i] for i in runs if endpoint_constraints[i]['feasible'] != 'crashed']
plt.hist(x,int(np.ceil(iters**0.5)), normed=None, facecolor='purple', edgecolor='black', alpha=1.0) 
plt.xlabel('tf [min]')
plt.ylabel('relative frequency')
plt.figure(4).savefig(path + 'tf.pdf')

# compute average tf
tf_bar = sum(tf[i] for i in runs if endpoint_constraints[i]['feasible'] != 'crashed')/sum(1 for i in runs if endpoint_constraints[i]['feasible'] != 'crashed')
endpoint_constraints['tf_avg'] = tf_bar

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

t = {}
T = {}
Tad = {}
for i in path_constraints: # loop over all runs
    if path_constraints[i] =='error':
        continue
    T[i] = []
    t[i] = []
    Tad[i] = []
    #T[i].append(path_constraints[i]['T',(1,(0,))])
    for fe in range(1,25):
        for cp in range(1,4):        
            T[i].append(path_constraints[i]['T',(fe,(cp,))])
            Tad[i].append(path_constraints[i]['Tad',(fe,(cp,))] if path_constraints[i]['Tad',(fe,(cp,))] > 0.0 else None)
            if fe > 1:
                t[i].append(t[i][-cp]+path_constraints[i]['tf',(fe,cp)])
            else:
                t[i].append(path_constraints[i]['tf',(fe,cp)])
    
    
max_tf = max([tf[i] for i in tf if endpoint_constraints[i]['feasible'] != 'crashed'])    
plt.figure(5)
for i in Tad:
    plt.plot(t[i],Tad[i], color='grey')
plt.plot([0,max_tf],[4.4315,4.4315], color='red', linestyle='dashed')
plt.plot()
plt.figure(5).savefig(path+'Tad.pdf')
plt.xlabel('t [min]')
plt.ylabel('Tad')
    
plt.figure(6)
for i in Tad:
    #tt = [0]
    #for k in t[i]:
    #    tt.append(k)
    plt.plot(t[i],T[i], color='grey')
#plt.plot([0,max_tf],[1.43403,1.43403], color='red', linestyle='dashed')
plt.plot([0,max_tf],[4.2315,4.2315], color='red', linestyle='dashed')
plt.plot([0,max_tf],[3.7315,3.7315], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('T')
plt.figure(6).savefig(path+'T.pdf')

# load results into file using pickle
#f = open('sampling_results.pckl', 'rb')
#obj = pickle.load(f)
#f.close()

# post processing
# get max violation
#max_vio = {}
#for constraint in endpoint_constraints[0]:
#        max_vio[constraint] = 0
#        if constraint == 'feasible':
#            continue    
#        for i in range(iters):
#            curr_vio = endpoint_constraints[i][constraint]
#            if max_vio[constraint] > curr_vio:
#                max_vio[constraint] = curr_vio
#            else:
#                continue
    
## constraint boxes
#box = {}
#counts = {}
#for constraint in endpoint_constraints[0]:
#    counts[constraint] = 0 # initialize counts
#    for i in range(intervalls):
#        box[constraint,i] = max_vio[constraint] * i/(intervalls-1)
#        
#fes = 0
## problem is feasible
#for i in range(iters):
## problem is feasible
#    if endpoint_constraints[i]['feasible'] == True:
#        fes += 1
#    for constraint in endpoint_constraints[0]:
#        for k in range(intervalls):
#            if box[constraint,k] <= endpoint_constraints[i][constraint]:
#                box[k] += 1
#                break

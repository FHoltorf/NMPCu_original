#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:39:42 2017

@author: flemmingholtorf
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from main.MC_sampling.run_MHE_asNMPC import *

# inputs
sample_size = 100
# specifiy directory where to save the resulting files
path = 'results/online_estimation/' 
# colors
color = ['green','red','blue']
tf = {}
endpoint_constraints = {}
path_constraints = {}
# run 100 batches and save the endpoint constraint violation
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
        tf[i],endpoint_constraints[i],path_constraints[i] = run()
    except ValueError:
        tf[i],endpoint_constraints[i],path_constraints[i] = 'error', {'epc_PO_ptg': 'error', \
                                   'epc_mw': 'error', \
                                   'epc_unsat': 'error'}, 'error'
    feasible = True
    for constraint in endpoint_constraints[i]:
        if endpoint_constraints[i][constraint] == 'error':
            feasible = 'crashed'
        elif endpoint_constraints[i][constraint] < 0:
            feasible = False
            break
    endpoint_constraints[i]['feasible'] = feasible

constraint_name = []

for constraint in endpoint_constraints[0]:
    if constraint == 'feasible':
        continue
    constraint_name.append(constraint)

unit = {'epc_PO_ptg' : ' [PPM]', 'epc_unsat' : ' [mol/g PO]', 'epc_mw' : ' [g/mol]'}

# enpoint constraints 
for k in range(3):
    color[k]
    x = [endpoint_constraints[i][constraint_name[k]] for i in range(sample_size) if endpoint_constraints[i][constraint_name[k]] != 'error']
    plt.figure(k)
    plt.hist(x,int(ceil(sample_size**0.5)), normed=None, facecolor=color[k], edgecolor='black', alpha=1.0) 
    plt.xlabel(constraint_name[k] + unit[constraint_name[k]])
    plt.ylabel('relative frequency')
    plt.figure(k).savefig(path + constraint_name[k] +'.pdf')
fes = 0
infes = 0

for i in range(sample_size):
    # problem is feasible
    if endpoint_constraints[i]['feasible'] == True:
        fes += 1
    elif endpoint_constraints[i]['feasible'] == False:
        infes += 1
sizes = [fes, infes, sample_size-fes-infes]

plt.figure(3)
plt.axis('equal')
explode = (0.0, 0.1, 0.0) 
wedges= plt.pie(sizes,explode,labels=['feasible','infeasible','crashed'], autopct='%1.1f%%',shadow=True)
for w in wedges[0]:
    w.set_edgecolor('black')
plt.figure(3).savefig(path + 'feas.pdf')
# save results to file using pickle

# compute average tf
tf_bar = sum(tf[i] for i in range(sample_size) if endpoint_constraints[i]['feasible'] != 'crashed')/sum(1 for i in range(sample_size) if endpoint_constraints[i]['feasible'] != 'crashed')
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

# path constraints
heat_removal = {}
t = {}
Tad = {}
for i in path_constraints: # loop over all runs
    if path_constraints[i] =='error':
        continue
    t_aux = []
    heat_removal[i] = []
    t[i] = []
    Tad[i] = []
    for fe in range(1,25):
        for cp in range(0,4):        
            heat_removal[i].append(path_constraints[i]['heat_removal',(fe,(cp,))])
            Tad[i].append(path_constraints[i]['Tad',(fe,(cp,))])
            t_aux.append(path_constraints[i]['tf',(fe,cp)])
    t[i] = [sum(t_aux[i] for i in range(k)) for k in range(len(t_aux))]   
    
plt.figure(4)
for i in Tad:
    plt.plot(t[i],Tad[i], color='grey')
plt.figure(4).savefig(path+'Tad.pdf')
plt.xlabel('t [min]')
plt.ylabel('Tad')
    
plt.figure(5)
for i in Tad:
    plt.plot(t[i],heat_removal[i], color='grey')
plt.xlabel('t [min]')
plt.ylabel('heat_removal')
plt.figure(4).savefig(path+'heat_removal.pdf')


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
#        for i in range(sample_size):
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
#for i in range(sample_size):
## problem is feasible
#    if endpoint_constraints[i]['feasible'] == True:
#        fes += 1
#    for constraint in endpoint_constraints[0]:
#        for k in range(intervalls):
#            if box[constraint,k] <= endpoint_constraints[i][constraint]:
#                box[k] += 1
#                break

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:54:57 2018

@author: flemmingholtorf
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

path = 'results/multimodel/' 
color = ['green','red','blue']

f = open(path + 'epc.pckl','rb')
endpoint_constraints = pickle.load(f)
f.close()

f = open(path + 'final_times.pckl','rb')
tf = pickle.load(f)
f.close()

f = open(path + 'path_constraints.pckl','rb')
path_constraints = pickle.load(f)
f.close()


constraint_name = []
iters = len(tf)
for constraint in endpoint_constraints[0]:
    if constraint == 'feasible':
        continue
    constraint_name.append(constraint)

unit = {'epc_PO_ptg' : ' [PPM]', 'epc_unsat' : ' [mol/g PO]', 'epc_mw' : ' [g/mol]'}

# enpoint constraints 
for k in range(3):
    color[k]
    x = [endpoint_constraints[i][constraint_name[k]] for i in range(iters) if endpoint_constraints[i][constraint_name[k]] != 'error']
    # compute standard deviation
    std = np.std(x) 
    mu = np.mean(x)
    # remove outliers (not in interval +-3 x std)
    x = [i for i in x if i >= mu-3*std and i <= mu+3*std]
    plt.figure(k)
    plt.hist(x, 'auto', normed=1, facecolor=color[k], edgecolor='black', alpha=1.0)
    plt.xlabel(constraint_name[k] + unit[constraint_name[k]])
    plt.ylabel('relative frequency')
    plt.figure(k).savefig(path + constraint_name[k] +'.pdf')


# compute final time histogram
plt.figure(4)
x = [tf[i] for i in range(iters) if endpoint_constraints[i]['feasible'] != 'crashed']
#plt.hist(x,int(np.ceil(iters**0.5)), normed=None, facecolor='purple', edgecolor='black', alpha=1.0) 
plt.hist(x,'auto', normed=1, facecolor='purple', edgecolor='black', alpha=1.0) 
plt.xlabel('tf [min]')
plt.ylabel('relative frequency')
plt.figure(4).savefig(path + 'tf.pdf')

# compute average tf
tf_bar = sum(tf[i] for i in range(iters) if endpoint_constraints[i]['feasible'] != 'crashed')/sum(1 for i in range(iters) if endpoint_constraints[i]['feasible'] != 'crashed')
endpoint_constraints['tf_avg'] = tf_bar


# path constraints
heat_removal = {}
t = {}
Tad = {}
for i in path_constraints: # loop over all runs
    if path_constraints[i] =='error':
        continue
    heat_removal[i] = []
    t[i] = []
    Tad[i] = []
    for fe in range(1,25):
        for cp in range(1,4):        
            heat_removal[i].append(path_constraints[i]['heat_removal',(fe,(cp,))])
            Tad[i].append(path_constraints[i]['Tad',(fe,(cp,))])
            if fe > 1:
                t[i].append(t[i][-cp]+path_constraints[i]['tf',(fe,cp)])
            else:
                t[i].append(path_constraints[i]['tf',(fe,cp)])
    
    
max_tf = max([tf[i] for i in tf if endpoint_constraints[i]['feasible'] != 'crashed'])    
plt.figure(5)
for i in Tad:
    plt.plot(t[i],Tad[i], color='grey')
plt.plot([0,max_tf],[4.6315,4.6315], color='red', linestyle='dashed')
plt.plot()
plt.figure(5).savefig(path+'Tad.pdf')
plt.xlabel('t [min]')
plt.ylabel('Tad')

# plot the enclosing hull
# much much more difficult than i thought!
plt.figure(6)
for i in Tad:
    plt.plot(t[i],heat_removal[i], color='grey')
plt.plot([0,max_tf],[1.43403,1.43403], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('heat_removal')
plt.figure(6).savefig(path+'heat_removal.pdf')


fes = 0
infes = 0

for i in range(iters):
    # problem is feasible
    if endpoint_constraints[i]['feasible'] == True:
        fes += 1
    elif endpoint_constraints[i]['feasible'] == False:
        infes += 1
sizes = [fes, infes, iters-fes-infes]

plt.figure(7)
plt.axis('equal')
explode = (0.0, 0.1, 0.0) 
wedges= plt.pie(sizes,explode,labels=['feasible','infeasible','crashed'], autopct='%1.1f%%',shadow=True)
for w in wedges[0]:
    w.set_edgecolor('black')
plt.figure(7).savefig(path + 'feas.pdf')
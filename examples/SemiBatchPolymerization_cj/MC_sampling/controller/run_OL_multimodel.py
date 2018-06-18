#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:54:54 2017

@author: flemmingholtorf
"""
from __future__ import print_function
from main.dync.MHEGen import msMHEGen
from main.mods.SemiBatchPolymerization_cj.mod_class_multistage_cj import SemiBatchPolymerization_multistage
from main.mods.SemiBatchPolymerization_cj.mod_class_cj import SemiBatchPolymerization
from main.examples.SemiBatchPolymerization_cj.noise_characteristics_cj import *
import sys ,pickle
import numpy as np
import matplotlib.pyplot as plt


path = 'results/'  
kA = np.array([-0.2,-0.1,0.0,0.1,0.2])#np.linspace(-0.2,0.2,num=4)
Ap = np.array([-0.2,-0.1,0.0,0.1,0.2])#np.linspace(-0.2,0.2,num=4)
Ai = np.array([-0.2,-0.1,0.0,0.1,0.2])#np.linspace(-0.2,0.2,num=4)
Ap, Ai, kA = np.meshgrid(kA, Ai, Ap)
i = 0
scenarios = {}
for j in range(len(kA)):
    for k in range(len(Ai)):
        for l in range(len(Ap)):
            scenarios[i] = {('A',('p',)):Ap[j][k][l],('A',('i',)):Ai[j][k][l],('kA',()):kA[j][k][l]}
            i += 1
            
sample_size = len(scenarios)

x_noisy = ["PO","MX","MY","Y","W","T"] # all the states are noisy  
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)],"T":[()],"T_cw":[()]}
p_noisy = {"A":[('p',),('i',)],'kA':[()]}
u = ["u1", "u2"]
u_bounds = {"u1": (-5.0, 5.0), "u2": (0.0, 3.0)} 

y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()],'T':[()]}
nfe = 24
ncp = 3
tf_bounds = [10.0*24.0/nfe, 30.0*24.0/nfe]

pc = ['Tad','T']

# scenario_tree
st = {} # scenario tree : {parent_node, scenario_number on current stage, base node (True/False), scenario values {'name',(index):value}}
s_max = 9
nr = 1
alpha = 0.2
for i in range(1,nfe+1):
    if i < nr + 1:
        for s in range(1,s_max**i+1):
            if s%s_max == 1:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),True,{('A',('p',)):1.0,('A',('i',)):1.0,('kA',()):1.0}) 
            elif s%s_max == 2:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
            elif s%s_max == 3:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
            elif s%s_max == 4:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
            elif s%s_max == 5:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
            elif s%s_max == 6:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
            elif s%s_max == 7:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
            elif s%s_max == 8:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
            else:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
    else:
        for s in range(1,s_max**nr+1):
            st[(i,s)] = (i-1,s,True,st[(i-1,s)][3])

sr = s_max**nr

e = msMHEGen(d_mod=SemiBatchPolymerization_multistage,
           d_mod_mhe = SemiBatchPolymerization,
           y=y_vars,
           x_noisy=x_noisy,
           x=x_vars,
           p_noisy=p_noisy,
           u=u,
           u_bounds = u_bounds,
           tf_bounds = tf_bounds,
           scenario_tree = st,
           robust_horizon = nr,
           s_max = sr,
           noisy_inputs = False,
           noisy_params = False,
           adapt_params = False,
           update_scenario_tree = False,
           confidence_threshold = alpha,
           robustness_threshold = 0.05,
           estimate_acceptance = 10000,
           process_noise_model = 'params',
           obj_type='economic',
           nfe_t=nfe,
           ncp_t=ncp,
           control_displacement_penalty = True,
           path_constraints=pc)

###############################################################################
###                                     NMPC
###############################################################################
e.recipe_optimization(multimodel=True)
e.set_reference_state_trajectory(e.get_state_trajectory(e.recipe_optimization_model))
e.set_reference_control_profile(e.get_control_profile(e.recipe_optimization_model))

endpoint_constraints, path_constraints = e.open_loop_simulation(sample_size=sample_size, parameter_scenario=scenarios)
for i in range(sample_size):
    feasible = True
    for constraint in endpoint_constraints[i]:
        if endpoint_constraints[i] == 'error':
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
color = ['green','red','blue']
for k in range(3):
    color[k]
    x = [endpoint_constraints[i][constraint_name[k]] for i in range(sample_size) if endpoint_constraints[i] != 'error']
    plt.figure(k)
    plt.hist(x,int(np.ceil(sample_size**0.5)), normed=None, facecolor=color[k], edgecolor='black', alpha=1.0) 
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

T = {}
t = {}
Tad = {}
for i in path_constraints: # loop over all runs
    if path_constraints[i] =='error':
        continue
    T[i] = []
    t[i] = []
    Tad[i] = []
    for fe in range(1,25):
        for cp in range(1,4):        
            T[i].append(path_constraints[i]['T',(fe,(cp,))]*1e2)
            Tad[i].append(path_constraints[i]['Tad',(fe,(cp,))]*1e2)
            if fe > 1:
                t[i].append(t[i][-cp]+path_constraints[i]['tf',(fe,cp)])
            else:
                t[i].append(path_constraints[i]['tf',(fe,cp)])

fig,ax = plt.subplots()
for i in Tad:
    ax.plot(t[i],Tad[i], color='grey')
ax.plot([0,t[1][-1]],[4.4315e2,4.4315e2], color='red', linestyle='dashed')
ax.set_xlabel(r'$t$ [min]')
ax.set_ylabel(r'$T_{ad}$ [K]')
ax.tick_params(axis='both',direction='in')
fig.savefig(path + 'Tad.pdf')

fig,ax = plt.subplots()
for i in Tad:
    ax.plot(t[i],T[i], color='grey')
ax.plot([0,t[1][-1]],[4.2315e2,4.2315e2], color='red', linestyle='dashed')
ax.plot([0,t[1][-1]],[3.7315e2,3.7315e2], color='red', linestyle='dashed')
ax.set_xlabel(r'$t$ [min]')
ax.set_ylabel(r'$T$ [K]')
ax.tick_params(axis='both',direction='in')
fig.savefig(path + 'T.pdf')


f = open(path + 'epc.pckl','wb')
pickle.dump(endpoint_constraints, f)
f.close()

f = open(path + 'path_constraints.pckl','wb')
pickle.dump(path_constraints,f)
f.close()


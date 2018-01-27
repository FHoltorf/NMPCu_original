#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:54:54 2017

@author: flemmingholtorf
"""

from __future__ import print_function
from pyomo.environ import *
# from nmpc_mhe.dync.MHEGen import MheGen
from main.dync.MHEGen_adjusted import MheGen
from main.mods.mod_class import *
import sys
import itertools, sys
import numpy as np
from main.noise_characteristics import *
import pickle
import matplotlib.pyplot as plt

# redirect system output to a file:
#sys.stdout = open('consol_output','w')

states = ["PO", "MX","X","MY","Y","W","m_tot","PO_fed"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
x_noisy = ["PO", "MX","X","MY","Y","W","m_tot","PO_fed"] # all the states are noisy
u = ["u1", "u2"]
u_bounds = {"u1": (373.15, 443.15), "u2": (0, 3.0)} # 14.5645661157
x_vars = {"PO":[()],"X":[()], "Y":[()], "W":[()], "m_tot":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)]}

#y_vars = x_vars 
#y = ["PO","X","m_tot","Y","W","PO_fed","MX","MY"] # only these variables get measured (note: have to be differential states)
#y_vars = {"PO":[()],"X":[()],"m_tot":[()],"Y":[()],"W":[()],"PO_fed":[()],"MX":[(0,),(1,)],"MY":[()]}
y = ["m_tot","MW","PO"] # only these variables get measured (note: have to be differential states)
y_vars = {"m_tot":[()],"MW":[()],"PO":[()]}
pc = ['Tad','heat_removal']
sample_size = 100 
color = ['green','red','blue']
# scenario_tree

e = MheGen(d_mod=SemiBatchPolymerization,
           d_mod_mhe=SemiBatchPolymerization,
           noisy_inputs = False,
           y=y,
           x_noisy=x_noisy,
           y_vars=y_vars,
           x_vars=x_vars,
           states=states,
           u=u,
           u_bounds=u_bounds,
           diag_QR=True,
           nfe_t=24,
           del_ics=False,
           noisy_params=True,
           path_constraints=pc)
e.recipe_optimization()
e.set_reference_state_trajectory(e.get_state_trajectory(e.recipe_optimization_model))
e.set_reference_control_trajectory(e.get_control_trajectory(e.recipe_optimization_model))

endpoint_constraints, path_constraints = e.open_loop_simulation(sample_size=sample_size,disturbances = v_disturbances, initial_disturbance=v_init, parameter_disturbance=v_param)
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

for k in range(3):
    color[k]
    x = [endpoint_constraints[i][constraint_name[k]] for i in range(sample_size) if endpoint_constraints[i] != 'error']
    plt.figure(k)
    plt.hist(x,int(ceil(sample_size**0.5)), normed=None, facecolor=color[k], edgecolor='black', alpha=1.0) 
    plt.xlabel(constraint_name[k] + unit[constraint_name[k]])
    plt.ylabel('relative frequency')
    plt.figure(k).savefig(constraint_name[k] +'.pdf')
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
plt.figure(3).savefig('feas.pdf')

f = open('sampling_results.pckl','wb')
pickle.dump(endpoint_constraints, f)
f.close()

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

plt.figure(5)
for i in Tad:
    plt.plot(t[i],Tad[i], color='grey')
plt.plot([0,t[1][-1]],[4.6315,4.6315], color='red', linestyle='dashed')
plt.plot()
plt.figure(5).savefig('Tad.pdf')
plt.xlabel('t [min]')
plt.ylabel('Tad')
    
plt.figure(6)
for i in Tad:
    plt.plot(t[i],heat_removal[i], color='grey')
plt.plot([0,t[1][-1]],[1.43403,1.43403], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('heat_removal')
plt.figure(6).savefig('heat_removal.pdf')

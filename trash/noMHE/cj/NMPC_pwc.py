#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:51:51 2017

@author: flemmingholtorf
"""
#### 

from __future__ import print_function
from pyomo.environ import *
# from nmpc_mhe.dync.MHEGen import MheGen
from main.dync.MHEGen_adjusted import MheGen
from main.mods.cj.mod_class_cj_pwc import *
import sys
import itertools, sys
import numpy as np
import matplotlib.pyplot as plt
from main.noise_characteristics_cj import *
import numpy.linalg as linalg
from scipy.stats import chi2
from copy import deepcopy
#redirect system output to a file:
#sys.stdout = open('consol_output.txt','w')


###############################################################################
###                               Specifications
###############################################################################

# all states + states that are subject to process noise (directly drawn from e.g. a gaussian distribution)
states = ["PO","MX","MY","Y","W","PO_fed","T"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
x_noisy = ["PO","MX","MY","Y","W","PO_fed","T"] # all the states are noisy  
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)], "T":[()]}
p_noisy = {"A":['p','i']}
u = ["u1", "u2"]
u_bounds = {"u1": (3.2315,4.4315), "u2": (0, 3.0)} # 14.5645661157

# measured variables
#y = {"heat_removal","Y","PO", "W", "MY", "MX", "MW","m_tot"}
#y_vars = {"heat_removal":[()],"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()]}
nfe = 24

pc = ['Tad','T']

e = MheGen(d_mod=SemiBatchPolymerization,
           x_noisy=x_noisy,
           x_vars=x_vars,
           states=states,
           p_noisy=p_noisy,
           u=u,
           noisy_inputs = False,
           noisy_params = False,
           adapt_params = False,
           u_bounds=u_bounds,
           diag_QR=True,
           nfe_t=nfe,
           del_ics=False,
           sens=None,
           obj_type='economic',
           path_constraints=pc)

e.delta_u = True
###############################################################################
###                                     NMPC
###############################################################################
e.recipe_optimization()
e.set_reference_state_trajectory(e.get_state_trajectory(e.recipe_optimization_model))
e.set_reference_control_trajectory(e.get_control_trajectory(e.recipe_optimization_model))
e.generate_state_index_dictionary()
e.create_enmpc() # with tracking-type regularization
#e.load_reference_trajectories()

k = 1
for i in range(1,nfe):
    print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
    if i == 1:
        e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call = True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.cycle_nmpc(e.store_results(e.recipe_optimization_model))
    else:
        e.plant_simulation(e.store_results(e.olnmpc),disturbance_src="parameter_noise",parameter_disturbance=v_param)
        e.cycle_nmpc(e.store_results(e.olnmpc))   

    e.cycle_ics_mhe(nmpc_as=False,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc

    #e.load_reference_trajectories() # loads the reference trajectory in olnmpc problem (for regularization)
    #e.set_regularization_weights(R_w=0.0,Q_w=0.0,K_w=0.0) # R_w controls, Q_w states, K_w = control steps
    e.solve_olnmpc() # solves the olnmpc problem
    e.olnmpc.write_nl()
    
    e.cycle_iterations()
    k += 1

    if  e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.plant_trajectory[i,'solstat'] != ['ok','optimal']:
        break
    
# simulate the last step too
e.plant_simulation_model.troubleshooting()

for i in range(1,k):
    print('iteration: %i' % i)
    print('open-loop optimal control: ', end='')
    print(e.nmpc_trajectory[i,'solstat'],e.nmpc_trajectory[i,'obj_value'])
    print('constraint inf: ', e.nmpc_trajectory[i,'eps'])
    print('plant: ',end='')
    print(e.plant_trajectory[i,'solstat'])



e.plant_simulation(e.store_results(e.olnmpc))


###############################################################################
####                        plot results comparisons   
###############################################################################

t_traj_nmpc = np.array([e.nmpc_trajectory[i,'tf'] for i in range(1,k)])
t_traj_sim = np.array([e.nmpc_trajectory[i,'tf'] for i in range(1,k+1)])
plt.figure(1)
tf = e.get_tf()
t = []
t.append(tf)
for i in range(1,nfe):
    aux = t[i-1] + tf
    t.append(aux)
#
l = 0

#moments
moment = ['MX']
for m in moment:
    for j in range(0,2):
        state_traj_ref = np.array([e.reference_state_trajectory[(m,(i,3,j))] for i in range(1,nfe+1)]) 
        state_traj_nmpc = np.array([e.nmpc_trajectory[i,(m,(j,))] for i in range(1,k)])
        state_traj_sim = np.array([e.plant_trajectory[i,(m,(j,))] for i in range(1,k+1)])
        plt.figure(l)
        plt.plot(t,state_traj_ref, label = "reference")
        plt.plot(t_traj_nmpc,state_traj_nmpc, label = "mhe/nmpc")
        plt.plot(t_traj_sim,state_traj_sim, label = "plant")
        plt.ylabel(m+str(j))
        plt.legend()
        l += 1

plots = [('Y',()),('PO',()),('PO_fed',()),('W',()),('MY',())]
for p in plots: 
    state_traj_ref = np.array([e.reference_state_trajectory[(p[0],(i,3))] for i in range(1,nfe+1)]) 
    state_traj_nmpc = np.array([e.nmpc_trajectory[i,p] for i in range(1,k)])
    state_traj_sim = np.array([e.plant_trajectory[i,p] for i in range(1,k+1)])    
    plt.figure(l)
    plt.plot(t,state_traj_ref, label = "reference")
    plt.plot(t_traj_nmpc,state_traj_nmpc, label = "mhe/nmpc")
    plt.plot(t_traj_sim,state_traj_sim, label = "plant")
    plt.legend()
    plt.ylabel(p[0])
    l += 1
    
plots = ['u1','u2']
t_traj_nmpc = [e.nmpc_trajectory[i,'tf'] for i in range(0,k+1)]
aux1 = []
aux2 = []
for i in range(1,len(t_traj_nmpc)):
    aux1.append(t_traj_nmpc[i-1])
    aux1.append(t_traj_nmpc[i])
    aux2.append((i-1)*tf)
    aux2.append(i*tf)
    
t = np.array(aux2)
t_traj_nmpc = np.array(aux1)
t_traj_sim = t_traj_nmpc 
for b in plots:
    aux_ref = []
    aux_nmpc = []
    aux_sim = []
    for i in range(1,k+1):
        for z in range(2):
            aux_ref.append(e.reference_control_trajectory[b,i])
            aux_nmpc.append(e.nmpc_trajectory[i,b])
            aux_sim.append(e.plant_trajectory[i,b])
    control_traj_ref = np.array(aux_ref)
    control_traj_nmpc = np.array(aux_nmpc)
    control_traj_sim = np.array(aux_sim)
    plt.figure(l)
    plt.plot(t,control_traj_ref, label = "reference")
    plt.plot(t_traj_nmpc,control_traj_nmpc, label = "predicted")
    plt.plot(t_traj_sim,control_traj_sim, label = "SBU")
    plt.legend()
    plt.ylabel(b)
    l += 1
    
e.plant_simulation_model.check_feasibility(display=True)

###############################################################################
###         Plotting path constraints
###############################################################################

l += 1
heat_removal = {}
t = {}
Tad = {}
path_constraints = {}
path_constraints[0] = e.pc_trajectory
for i in range(1): # loop over all runs
    if path_constraints[i] =='error':
        continue
    heat_removal[i] = []
    t[i] = []
    Tad[i] = []
    for fe in range(1,25):
        for cp in range(1,4):        
            heat_removal[i].append(path_constraints[i]['T',(fe,(cp,))])
            Tad[i].append(path_constraints[i]['Tad',(fe,(cp,))])
            if fe > 1:
                t[i].append(t[i][-cp]+path_constraints[i]['tf',(fe,cp)])
            else:
                t[i].append(path_constraints[i]['tf',(fe,cp)])
    
    
max_tf = max(t[0])   
plt.figure(l)
for i in Tad:
    plt.plot(t[i],Tad[i], color='grey')
plt.plot([0,max_tf],[4.4315,4.4315], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('Tad [K]')
    
l += 1
plt.figure(l)
for i in heat_removal:
    plt.plot(t[i],heat_removal[i], color='grey')
plt.plot([0,max_tf],[443.15/100,443.15/100], color='red', linestyle='dashed')
plt.plot([0,max_tf],[373.15/100,373.15/100], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('T [K]')


print('MULTISTAGE NMPC')
print('OPTIONS:')
print('measured vars ', e.y)
print('state vars ', e.states)
print('pars estimated online ', e.noisy_params)
print('pars adapted ', e.adapt_params)
print('update ST ', e.update_scenario_tree)
print('confidence threshold ', e.confidence_threshold)
print('robustness threshold ', e.robustness_threshold)
print('estimate_acceptance ', e.estimate_acceptance)

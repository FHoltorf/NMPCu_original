#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:51:51 2017

@author: flemmingholtorf
"""
####
from __future__ import print_function
from pyomo.environ import *
from scipy.stats import chi2
from copy import deepcopy
from main.dync.MHEGen_adjusted import MheGen
from main.mods.final_pwa.mod_class_cj_pwa_robust_optimal_control import *
from main.noise_characteristics_cj import * 
import itertools, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

#redirect system output to a file:
#sys.stdout = open('consol_output.txt','w')


###############################################################################
###                               Specifications
###############################################################################
states = ["PO","MX","MY","Y","W","PO_fed","T","T_cw"]
x_noisy = ["PO","MX","MY","Y","W","T"]
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)], "T":[()], "T_cw":[()]}

u = ["u1", "u2"]
u_bounds = {"u1": (-5.0, 5.0), "u2": (0.0, 3.0)} 

y = {"Y","PO", "W", "MY", "MX", "MW","m_tot",'T'}
y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()],'T':[()]}
nfe = 24
tf_bounds = [10.0*24.0/nfe, 30.0*24.0/nfe]



cons = ['mw','mw_ub','PO_ptg','unsat','temp_b','T_min','T_max']
pc = ['Tad','T']
p_noisy = {"A":[('p',),('i',)],'kA':[()]}
alpha = {('A',('p',)):0.2,('A',('i',)):0.2,('kA',()):0.2,
          ('T_ic',()):0.005,
          ('MX_ic',(1,)):0.005,
          ('PO_ic',()):0.02,
          ('MY_ic',()):0.01}

scenario = {('A', ('i',)): -0.2, ('A', ('p',)): -0.2, ('kA', ()): -0.1}
e = MheGen(d_mod=SemiBatchPolymerization,
           linapprox = True,
           alpha = alpha,
           x_noisy=x_noisy,
           x_vars=x_vars,
           y=y,
           y_vars=y_vars,
           states=states,
           p_noisy=p_noisy,
           u=u,
           noisy_inputs = False,
           noisy_params = False,
           adapt_params = False,
           process_noise_model = 'params_bias',
           u_bounds=u_bounds,
           tf_bounds = tf_bounds,
           diag_QR=False,
           nfe_t=nfe,
           del_ics=False,
           sens=None,
           obj_type='tracking',
           path_constraints=pc)
delta_u = True

###############################################################################
###                                     NMPC
###############################################################################
model = e.recipe_optimization(cons=cons,eps=1e-1)
e.set_reference_state_trajectory(e.get_state_trajectory(e.recipe_optimization_model))
e.set_reference_control_trajectory(e.get_control_trajectory(e.recipe_optimization_model))
e.generate_state_index_dictionary()
e.create_nmpc() # with tracking-type regularization
e.load_reference_trajectories()

k = 1
for i in range(1,nfe):
    print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
    e.create_mhe()
    if i == 1:
        #e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call = True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call = True,disturbance_src = "parameter_scenario",scenario=scenario)
        e.set_measurement_prediction(e.store_results(e.recipe_optimization_model))
        e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
        e.cycle_mhe(e.store_results(e.recipe_optimization_model),mcov,qcov,ucov,p_cov=pcov) #adjusts the mhe problem according to new available measurements
        e.cycle_nmpc(e.store_results(e.recipe_optimization_model))
    else:
        #e.plant_simulation(e.store_results(e.olnmpc),disturbance_src="parameter_noise",parameter_disturbance=v_param)
        e.plant_simulation(e.store_results(e.olnmpc),disturbance_src="parameter_scenario",scenario=scenario)
        e.set_measurement_prediction(e.store_results(e.plant_simulation_model))
        e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
        e.cycle_mhe(previous_mhe,mcov,qcov,ucov,p_cov=pcov) 
        e.cycle_nmpc(e.store_results(e.olnmpc))     

    # here measurement becomes available
    previous_mhe = e.solve_mhe(fix_noise=True) # solves the mhe problem
    # solve the advanced step problems
    e.cycle_ics_mhe(nmpc_as=False,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc

    e.load_reference_trajectories() # loads the reference trajectory in olnmpc problem (for regularization)
    e.set_regularization_weights(R_w=0.0,Q_w=0.0,K_w=1.0) # R_w controls, Q_w states, K_w = control steps
    model = e.solve_olrnmpc(cons=cons,eps=1e-4) # solves the olnmpc problem
    

    e.cycle_iterations()
    k += 1

    if  e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.nmpc_trajectory[i,'solstat_mhe'] != ['ok','optimal'] or \
        e.plant_trajectory[i,'solstat'] != ['ok','optimal']:
        break
    
for i in range(1,k):
    print('iteration: %i' % i)
    print('open-loop optimal control: ', end='')
    print(e.nmpc_trajectory[i,'solstat'],e.nmpc_trajectory[i,'obj_value'])
    print('constraint inf: ', e.nmpc_trajectory[i,'eps'])
    print('plant: ',end='')
    print(e.plant_trajectory[i,'solstat'])

e.plant_simulation(e.store_results(e.olnmpc))




# Uncertainty Realization
print('uncertainty realization')
for p in p_noisy:
    pvar_r = getattr(e.plant_simulation_model, p)
    pvar_m = getattr(e.recipe_optimization_model, p)
    for key in p_noisy[p]:
        if key != ():
            print('delta_p ',p,key,': ',(pvar_r[key].value-pvar_m[key].value)/pvar_m[key].value)
        else:
            print('delta_p ',p,key,': ',(pvar_r.value-pvar_m.value)/pvar_m.value)
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
        state_traj_ref = np.array([e.reference_state_trajectory[(m,(i,3,j,1))] for i in range(1,nfe+1)]) 
        state_traj_nmpc = np.array([e.nmpc_trajectory[i,(m,(j,1))] for i in range(1,k)])
        state_traj_sim = np.array([e.plant_trajectory[i,(m,(j,1))] for i in range(1,k+1)])
        plt.figure(l)
        plt.plot(t,state_traj_ref, label = "reference")
        plt.plot(t_traj_nmpc,state_traj_nmpc, label = "mhe/nmpc")
        plt.plot(t_traj_sim,state_traj_sim, label = "plant")
        plt.ylabel(m+str(j))
        plt.legend()
        l += 1

plots = [('Y',(1,)),('PO',(1,)),('PO_fed',(1,)),('W',(1,)),('MY',(1,)),('T_cw',(1,)),('T',(1,))]
#plots = [('PO',()),('T_cw',()),('T',())]
for p in plots: 
    state_traj_ref = np.array([e.reference_state_trajectory[(p[0],(i,3,1))] for i in range(1,nfe+1)]) 
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

# print uncertatinty realization


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
    for fe in range(1,k+1):
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
plt.plot([0,max_tf],[423.15/100,423.15/100], color='red', linestyle='dashed')
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

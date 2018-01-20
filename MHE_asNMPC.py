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
from main.mods.mod_class import *
import sys
import itertools, sys
import numpy as np
import matplotlib.pyplot as plt
from main.noise_characteristics import *
import numpy.linalg as linalg
from scipy.stats import chi2
from copy import deepcopy
#redirect system output to a file:
#sys.stdout = open('consol_output.txt','w')


###############################################################################
###                               Specifications
###############################################################################

# all states + states that are subject to process noise (directly drawn from e.g. a gaussian distribution)
states = ["PO","MX","MY","Y","W","PO_fed"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
x_noisy = ["PO","MX","MY","Y","W","PO_fed"] # all the states are noisy  
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)]}
p_noisy = {"A":['p','i']}
u = ["u1", "u2"]
u_bounds = {"u1": (373.15/1e2, 443.15/1e2), "u2": (0, 3.0)} # 14.5645661157

# measured variables
y = {"heat_removal","Y","PO", "W", "MY", "MX", "MW","m_tot"}
y_vars = {"heat_removal":[()],"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()]}
nfe = 24

pc = ['Tad','heat_removal']

e = MheGen(d_mod=SemiBatchPolymerization,
           y=y,
           x_noisy=x_noisy,
           y_vars=y_vars,
           x_vars=x_vars,
           states=states,
           p_noisy=p_noisy,
           u=u,
           noisy_inputs = False,
           noisy_params = True,
           adapt_params = False,
           u_bounds=u_bounds,
           diag_QR=True,
           nfe_t=nfe,
           del_ics=False,
           sens=None,
           path_constraints=pc)

e.delta_u = True
###############################################################################
###                                     NMPC
###############################################################################
e.recipe_optimization()
e.set_reference_state_trajectory(e.get_state_trajectory(e.recipe_optimization_model))
e.set_reference_control_trajectory(e.get_control_trajectory(e.recipe_optimization_model))
e.generate_state_index_dictionary()
e.create_nmpc() # with tracking-type regularization
e.load_reference_trajectories()

k = 1  

before = {}
after = {}
diff = {}
applied = {}
state_offset = {} 
curr_estate = {}
curr_pstate = {}
#try:
for i in range(1,nfe):
    print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
    e.create_mhe()
    if i == 1:
        e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call = True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.set_measurement_prediction(e.store_results(e.recipe_optimization_model))
        e.cycle_mhe(e.store_results(e.recipe_optimization_model),mcov,qcov,ucov) #adjusts the mhe problem according to new available measurements
        e.cycle_nmpc(e.store_results(e.recipe_optimization_model))
    else:
        e.plant_simulation(e.store_results(e.olnmpc),disturbance_src="parameter_noise",parameter_disturbance=v_param)
        e.set_measurement_prediction(e.store_results(e.forward_simulation_model))
        e.cycle_mhe(previous_mhe,mcov,qcov,ucov) 
        e.cycle_nmpc(e.store_results(e.olnmpc))   
    
    # solve the advanced step problems
    e.cycle_ics_mhe(nmpc_as=True,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc

    e.load_reference_trajectories() # loads the reference trajectory in olnmpc problem (for regularization)
    e.set_regularization_weights(R_w=0.0,Q_w=0.0,K_w=0.0) # R_w controls, Q_w states, K_w = control steps
    e.solve_olnmpc() # solves the olnmpc problem
    e.olnmpc.write_nl()
    
    # preparation for nmpc
    e.create_suffixes_nmpc()
    e.sens_k_aug_nmpc()
    
    # here measurement becomes available
    e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
    
    #solve mhe problem
    for z in range(i):
        e.lsmhe.wk_mhe[z,1].fix() # MX0
        e.lsmhe.wk_mhe[z,2].fix() # MX1
        e.lsmhe.wk_mhe[z,6].fix() #PO_fed 
        e.lsmhe.wk_mhe[z,0].fix() # PO
        e.lsmhe.wk_mhe[z,3].fix() #MY
        e.lsmhe.wk_mhe[z,5].fix() #W
        e.lsmhe.wk_mhe[z,4].fix() #Y
    e.solve_mhe(fix_noise=False) # solves the mhe problem
    previous_mhe = e.store_results(e.lsmhe)
    #e.compute_confidence_ellipsoid()
    
    # update state estimate 
    e.update_state_mhe() # can compute offset within this function by setting as_nmpc_mhe_strategy = True
    
    # compute fast update for nmpc
    state_offset[i], curr_pstate[i], curr_estate[i] = e.compute_offset_state(src_kind="estimated")
    
    before[i], after[i], diff[i], applied[i] = e.sens_dot_nmpc()   
    
    # forward simulation for next iteration
    e.forward_simulation()
    e.cycle_iterations()
    k += 1

    if  e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.nmpc_trajectory[i,'solstat_mhe'] != ['ok','optimal'] or \
        e.plant_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.simulation_trajectory[i,'solstat'] != ['ok','optimal']:
        with open("000aaa.txt","w") as f:
            f.write('plant :' + e.plant_trajectory[i,'solstat'][1] + '\n' \
                    + 'nmpc :' + e.nmpc_trajectory[i,'solstat'][1] + '\n' \
                    + 'simulation :' + e.simulation_trajectory[i,'solstat'][1])
        break
    
# simulate the last step too
e.plant_simulation_model.troubleshooting()

for i in range(1,k):
    print('iteration: %i' % i)
    print('open-loop optimal control: ', end='')
    print(e.nmpc_trajectory[i,'solstat'],e.nmpc_trajectory[i,'obj_value'])
    print('constraint inf: ', e.nmpc_trajectory[i,'eps'])
    print('mhe: ', end='')
    print(e.nmpc_trajectory[i,'solstat_mhe'], e.nmpc_trajectory[i, 'obj_value_mhe'])
    print('plant: ',end='')
    print(e.plant_trajectory[i,'solstat'])
    print('forward_simulation: ',end='')
    print(e.simulation_trajectory[i,'solstat'], e.simulation_trajectory[i,'obj_fun'])


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

state_offset_norm = []
diff_norm = []

for u in ['u1','u2']:    
    x = []
    for i in range(1,k):
        x.append(diff[i][u])
    l += 1
    plt.figure(l)
    plt.plot(x)

for p in [('Y',()),('PO',()),('PO_fed',()),('W',())]:
    x = []
    y = [] 
    z = []
    for i in range(1,k):
        x.append(curr_pstate[i][p])
        y.append(curr_estate[i][p])
        z.append(curr_pstate[i][p] - state_offset[i][p])
    l += 1
    plt.figure(l)
    plt.plot(x, label = 'predicted')
    plt.plot(y, label = 'estimated')
    plt.plot(z, label = 'estimated check')
    plt.ylabel(p[0])
    plt.legend()
    

# compute the confidence ellipsoids
# delta_theta^T*Vi*delta_theta = sigma^2*chi2(n_dof,confidence_level)
l += 1
plt.figure(l)

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
            heat_removal[i].append(path_constraints[i]['heat_removal',(fe,(cp,))])
            Tad[i].append(path_constraints[i]['Tad',(fe,(cp,))])
            if fe > 1:
                t[i].append(t[i][-cp]+path_constraints[i]['tf',(fe,cp)])
            else:
                t[i].append(path_constraints[i]['tf',(fe,cp)])
    
    
max_tf = max(t[0])   
plt.figure(l)
for i in Tad:
    plt.plot(t[i],Tad[i], color='grey')
plt.plot([0,max_tf],[4.6315,4.6315], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('Tad')
    
l += 1
plt.figure(l)
for i in heat_removal:
    plt.plot(t[i],heat_removal[i], color='grey')
plt.plot([0,max_tf],[1.43403,1.43403], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('heat_removal')


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

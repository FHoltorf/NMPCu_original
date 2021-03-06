#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:51:51 2017
@author: flemmingholtorf
"""
#### 
from __future__ import print_function
from pyomo.environ import *
from main.dync.MHEGen_multistage import MheGen
from main.mods.final_pwa.mod_class_endperiod import *
from main.mods.final_pwa.mod_class_cj_pwa import *
import sys
import itertools, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import chi2
from main.noise_characteristics_cj import *

# redirect system output to a file:
#sys.stdout = open('consol_output','w')

###############################################################################
###                               Specifications
###############################################################################
gamma = 0
#while(True):
#    gamma+=1
print('#'*15,gamma)
states = ["PO","MX","MY","Y","W","PO_fed","T","T_cw"]
x_noisy = ["PO","MX","MY","Y","W","T"]   
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)],"T":[()],"T_cw":[()]}
p_noisy = {"A":[('p',),('i',)],'kA':[()]}
u = ["u1", "u2"]
u_bounds = {"u1": (-5.0, 5.0), "u2": (0, 3.0)} 

cons = ['mw','unsat','PO_ptg','temp_b','T_max','T_min']#,'mw']#,'mw','mw_ub'
#cons = ['temp_b','T_min','T_max']
#y = {"Y","PO", "W", "MY", "MX", "MW","m_tot",'T'}
#y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()],'T':[()]}
y = {"Y","MY","PO",'T'}
y_vars = {"Y":[()],"MY":[()],"PO":[()],'T':[()]}

noisy_ics = {'PO_ic':[()],'T_ic':[()],'MY_ic':[()],'MX_ic':[(1,)]}
p_bounds = {('A', ('i',)):(-0.2,0.2),('A', ('p',)):(-0.2,0.2),('kA',()):(-0.2,0.2),
            ('PO_ic',()):(-0.02,0.02),('T_ic',()):(-0.005,0.005),
            ('MY_ic',()):(-0.01,0.01),('MX_ic',(1,)):(-0.005,0.005)}

nfe = 24
tf_bounds = [10.0*24.0/nfe, 30.0*24.0/nfe]

pc = ['Tad','T']

# scenario_tree
st = {} # scenario tree : {parent_node, scenario_number on current stage, base node (True/False), scenario values {'name',(index):value}}
s_max = 3
nr = 2
alpha = 0.2
alpha_1 = 0.2
dummy ={(1, 2): {('A', ('p',)): 1-alpha_1, ('kA', ()): 1-alpha, ('T_ic', ()): 1+0.005, ('A', ('i',)): 1-alpha, ('MY_ic', ()): 1+0.01, ('PO_ic', ()): 1+0.02}, 
         (1, 3): {('A', ('p',)): 1-alpha_1, ('kA', ()): 1+alpha, ('T_ic', ()): 1-0.005, ('A', ('i',)): 1-alpha, ('MY_ic', ()): 1-0.01, ('PO_ic', ()): 1+0.02}, 
         (1, 4): {('A', ('p',)): 1-alpha_1, ('kA', ()): 1-alpha, ('T_ic', ()): 1-0.005, ('A', ('i',)): 1-alpha, ('MY_ic', ()): 1-0.01, ('PO_ic', ()): 1+0.02}, 
         (1, 5): {('A', ('p',)): 1-alpha_1, ('kA', ()): 1-alpha, ('T_ic', ()): 1-0.005, ('A', ('i',)): 1+alpha, ('MY_ic', ()): 1-0.01, ('PO_ic', ()): 1+0.02}, 
         (1, 6): {('A', ('p',)): 1+alpha_1, ('kA', ()): 1+alpha, ('T_ic', ()): 1-0.005, ('A', ('i',)): 1-alpha, ('MY_ic', ()): 1-0.01, ('PO_ic', ()): 1+0.02}, 
         (1, 7): {('A', ('p',)): 1+alpha_1, ('kA', ()): 1+alpha, ('T_ic', ()): 1-0.005, ('A', ('i',)): 1+alpha, ('MY_ic', ()): 1-0.01, ('PO_ic', ()): 1+0.02}, 
         (1, 8): {('A', ('p',)): 1+alpha_1, ('kA', ()): 1-alpha, ('T_ic', ()): 1-0.005, ('A', ('i',)): 1-alpha, ('MY_ic', ()): 1-0.01, ('PO_ic', ()): 1+0.02}, 
         (1, 9): {('A', ('p',)): 1+alpha_1, ('kA', ()): 1-alpha, ('T_ic', ()): 1-0.005, ('A', ('i',)): 1+alpha, ('MY_ic', ()): 1-0.01, ('PO_ic', ()): 1+0.02}, 
         (2, 2): {('A', ('p',)): 1-alpha_1, ('A', ('i',)): 1-alpha, ('kA', ()): 1+alpha},
         (2, 3): {('A', ('p',)): 1-alpha_1, ('A', ('i',)): 1-alpha, ('kA', ()): 1-alpha},
         (2, 4): {('A', ('p',)): 1-alpha_1, ('A', ('i',)): 1-alpha, ('kA', ()): 1-alpha},
         (3, 2): {('A', ('p',)): 1-alpha_1, ('A', ('i',)): 1-alpha, ('kA', ()): 1+alpha},
         (3, 3): {('A', ('p',)): 1-alpha_1, ('A', ('i',)): 1-alpha, ('kA', ()): 1-alpha}}

for i in range(1,nfe+1):
    if i < nr + 1:
        for s in range(1,s_max**i+1):
            if s%s_max == 1:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),True,{('A',('p',)):1.0,('A',('i',)):1.0,('kA',()):1.0}) 
            else:
                scen = s%s_max if s%s_max != 0 else 3
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,dummy[(i,scen)])
    else:
        for s in range(1,s_max**nr+1):
            st[(i,s)] = (i-1,s,True,st[(i-1,s)][3])
            
#for i in range(1,nfe+1):
#    if i < nr + 1:
#        for s in range(1,s_max**i+1):
#            if s%s_max == 1:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),True,{('A',('p',)):1.0,('A',('i',)):1.0,('kA',()):1.0}) 
#            elif s%s_max == 2:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
#            elif s%s_max == 3:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
#            elif s%s_max == 4:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
#            elif s%s_max == 5:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
#            elif s%s_max == 6:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
#            elif s%s_max == 7:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
#            elif s%s_max == 8:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
#            else:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
#    else:
#        for s in range(1,s_max**nr+1):
#            st[(i,s)] = (i-1,s,True,st[(i-1,s)][3])

sr = s_max**nr

e = MheGen(d_mod=SemiBatchPolymerization_multistage,
           d_mod_mhe = SemiBatchPolymerization,
           y=y,
           y_vars=y_vars,
           x_noisy=x_noisy,
           x_vars=x_vars,
           p_noisy=p_noisy,
           uncertainty_set=p_bounds,
           states=states,
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
           process_noise_model = 'params_bias',
           confidence_threshold = alpha,
           robustness_threshold = 0.05,
           estimate_exceptance = 10000,
           obj_type='tracking',
           nfe_t=nfe,
           sens=None,
           diag_QR=False,
           del_ics=False,
           path_constraints=pc)

e.delta_u = True

###############################################################################
###                                     NMPC
###############################################################################
e.recipe_optimization()
e.set_reference_state_trajectory(e.get_state_trajectory(e.recipe_optimization_model))
e.set_reference_control_trajectory(e.get_control_trajectory(e.recipe_optimization_model))
e.generate_state_index_dictionary()
e.create_enmpc()

e.create_mhe()

k = 1 
for i in range(1,nfe):
    print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
    e.create_mhe()
    if i == 1:
        #e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call=True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call=True,disturbance_src = "parameter_scenario",scenario={('A',('p',)):-0.2,('A',('i',)):-0.2,('kA',()):-0.2})
        e.set_measurement_prediction(e.store_results(e.recipe_optimization_model)) # only required for asMHE
        e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
        e.cycle_mhe(e.store_results(e.recipe_optimization_model),mcov,qcov,ucov,p_cov=pcov, first_call=True) #adjusts the mhe problem according to new available measurements
        e.SBWCS_hyrec(epc=cons[:3], pc=cons[3:],par_bounds=p_bounds,crit='con',noisy_ics=noisy_ics)
        e.cycle_nmpc(e.store_results(e.recipe_optimization_model))
    else:
        #e.plant_simulation(e.store_results(e.olnmpc),disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.plant_simulation(e.store_results(e.olnmpc),disturbance_src = "parameter_scenario",scenario={('A',('p',)):-0.2,('A',('i',)):-0.2,('kA',()):-0.2})
        e.set_measurement_prediction(e.store_results(e.plant_simulation_model))
        e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)          
        e.cycle_mhe(previous_mhe,mcov,qcov,ucov,p_cov=pcov) # only required for asMHE   
        e.SBWCS_hyrec(epc=cons[:3], pc=cons[3:],par_bounds=p_bounds,crit='con',noisy_ics=noisy_ics)
        e.cycle_nmpc(e.store_results(e.olnmpc))   

    e.load_reference_trajectories()
    previous_mhe = e.solve_mhe(fix_noise=True)

        
    if e.update_scenario_tree:
        e.compute_confidence_ellipsoid()
    e.cycle_ics_mhe(nmpc_as=False,mhe_as=False)

    e.set_regularization_weights(K_w = 1.0, Q_w = 0.0, R_w = 0.0)
    
    e.solve_olnmpc()         
    e.cycle_iterations()
    k += 1
#    if i == 22:
#        sys.exit()

    #troubleshooting
    if  e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.nmpc_trajectory[i,'solstat_mhe'] != ['ok','optimal'] or \
        e.plant_trajectory[i,'solstat'] != ['ok','optimal']:
        sys.exit()
        break

# simulate the last step too

#e.forward_simulation_model.troubleshooting()
e.plant_simulation_model.troubleshooting()

for i in range(1,k):
    print('iteration: %i' % i)
    print('open-loop optimal control: ', end='')
    print(e.nmpc_trajectory[i,'solstat'],e.nmpc_trajectory[i,'obj_value'])
    print('constraint inf: ', e.nmpc_trajectory[i,'eps'])
    print('mhe: ', end='')
    print(e.nmpc_trajectory[i,'solstat_mhe'], e.nmpc_trajectory[i,'obj_value_mhe'])
    print('plant: ',end='')
    print(e.plant_trajectory[i,'solstat'])


e.plant_simulation(e.store_results(e.olnmpc))
e.plant_simulation_model.check_feasibility(display=True)

###############################################################################
####                        plot results comparisons   
###############################################################################

t_traj_nmpc = np.array([e.nmpc_trajectory[i,'tf'] for i in range(1,k)])
t_traj_sim = np.array([e.nmpc_trajectory[i,'tf'] for i in range(1,k+1)])
plt.figure(1)

t = e.get_tf(1)
#
l = 0

#moments
moment = ['MX']
for m in moment:
    for j in range(0,2):
        state_traj_ref = np.array([e.reference_state_trajectory[(m,(i,3,j,1))] for i in range(1,nfe+1)]) 
        state_traj_nmpc = np.array([e.nmpc_trajectory[i,(m,(j,))] for i in range(1,k)])
        state_traj_sim = np.array([e.plant_trajectory[i,(m,(j,))] for i in range(1,k+1)])
        plt.figure(l)
        plt.plot(t,state_traj_ref, label = "reference")
        plt.plot(t_traj_nmpc,state_traj_nmpc, label = "mhe/nmpc")
        plt.plot(t_traj_sim,state_traj_sim, label = "plant")
        plt.ylabel(m+str(j))
        plt.legend()
        l += 1

plots = [('Y',()),('PO',()),('PO_fed',()),('W',()),('MY',()),('T',())]
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
t = [0] + t 
aux1 = []
aux2 = []
for i in range(1,len(t_traj_nmpc)):
    aux1.append(t_traj_nmpc[i-1])
    aux1.append(t_traj_nmpc[i])
    aux2.append(t[i-1])
    aux2.append(t[i])
    
t = np.array(aux2)
t_traj_nmpc = np.array(aux1)
t_traj_sim = t_traj_nmpc 
for b in plots:
    aux_ref = []
    aux_nmpc = []
    aux_sim = []
    for i in range(1,k+1):
        for z in range(2):
            aux_ref.append(e.reference_control_trajectory[b,(i,1)]) # reference computed off-line/recipe optimization
            aux_nmpc.append(e.nmpc_trajectory[i,b]) # nmpc --> predicted one step ahead
            aux_sim.append(e.plant_trajectory[i,b]) # after advanced step --> one step ahead
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

###############################################################################
###         Plotting 1st Order Approximation of Confidence Region 
###############################################################################
#dimension = 2 # dimension n of the n x n matrix = #DoF
#rhs_confidence = chi2.isf(1.0-0.95,dimension) # 0.1**2*5% measurment noise, 95% confidence level, dimension degrees of freedo
#rows = {}
#Scaling = np.diag([329132.265895, 13886.7548015])
#for r in range(1,k):
#    A_dict = e.mhe_confidence_ellipsoids[r]
#    center = [0,0]
#    for m in range(dimension):
#        rows[m] = np.array([A_dict[(m,i)] for i in range(dimension)])
#    A = 1/rhs_confidence*np.dot(Scaling.transpose(), np.dot(np.array([np.array(rows[i]) for i in range(dimension)]),Scaling))
#    center = np.array([0]*dimension)
#    U, s, V = linalg.svd(A) # singular value decomposition 
#    radii = 1/np.sqrt(s) # length of half axes, V rotation
#    
#    # transform in polar coordinates for simpler waz of plotting
#    theta = np.linspace(0.0, 2.0 * np.pi, 100) # angle = idenpendent variable
#    x = radii[0] * np.sin(theta) # x-coordinate
#    y = radii[1] * np.cos(theta) # y-coordinate
#    for i in range(len(x)):
#        [x[i],y[i]] = np.dot([x[i],y[i]], V) + center
#    plt.plot(x,y, label = str(r))
#
#    
#    # plot half axis
#for p in range(dimension):
#    x = radii[p]*U[p][0]
#    y = radii[p]*U[p][1]
#    plt.plot([0,x],[0,y],color='red')
#plt.xlabel(r'$\Delta A_i$')
#plt.ylabel(r'$\Delta A_p$')


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

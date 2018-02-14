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
from main.mods.final.mod_class import *
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
# all states + states that are subject to process noise (directly drawn from e.g. a gaussian distribution)
states = ["PO","MX","MY","Y","W","PO_fed","T","T_cw"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
x_noisy = ["PO","MX","MY","Y","W","T","T_cw"] # all the states are noisy  
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)], "T":[()], "T_cw":[()]}
#p_noisy = {"A":[('p',),('i',)],'kA':[()],'Hrxn_aux':[('p',)]}
p_noisy = {"A":[('p',),('i',)],'kA':[()]}
u = ["u1", "u2"]
u_bounds = {"u1": (0.0,0.3), "u2": (0.0, 3.0)} 

y = {'T','T_cw','m_tot','W','MX',"Y","PO","MY"}
y_vars = {"Y":[()],"PO":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()],'T':[()],'T_cw':[()]}
#y = {"ByProd","PO",'T'}
#y_vars = {"ByProd":[()],"PO":[()],'T':[()]}

nfe = 24
tf_bounds = [10.0*24.0/nfe, 30.0*24.0/nfe]

pc = ['Tad','T']
e = MheGen(d_mod=SemiBatchPolymerization,
           x_noisy=x_noisy,
           x_vars=x_vars,
           y=y,
           y_vars=y_vars,
           states=states,
           p_noisy=p_noisy,
           u=u,
           noisy_inputs = False,
           noisy_params = True,
           adapt_params = True,
#           process_noise_model = 'params',
           u_bounds=u_bounds,
           tf_bounds = tf_bounds,
           diag_QR=True,
           nfe_t=nfe,
           del_ics=False,
           sens=None,
           obj_type='tracking',
           path_constraints=pc)
delta_u = True

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
for i in range(1,nfe):
    print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
    e.create_mhe()
    if i == 1:
        e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call = True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.set_measurement_prediction(e.store_results(e.recipe_optimization_model))
        e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
        e.cycle_mhe(e.store_results(e.recipe_optimization_model),mcov,qcov,ucov,p_cov=pcov) #adjusts the mhe problem according to new available measurements
        e.cycle_nmpc(e.store_results(e.recipe_optimization_model))
    else:
        e.plant_simulation(e.store_results(e.olnmpc),disturbance_src="parameter_noise",parameter_disturbance=v_param)
        e.set_measurement_prediction(e.store_results(e.forward_simulation_model))
        e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
        e.cycle_mhe(previous_mhe,mcov,qcov,ucov,p_cov=pcov) 
        e.cycle_nmpc(e.store_results(e.olnmpc))     
    sys.exit()
    # here measurement becomes available
    previous_mhe = e.solve_mhe(fix_noise=True) # solves the mhe problem
    #sys.exit()
    e.compute_confidence_ellipsoid()
    e.cycle_ics_mhe(nmpc_as=False,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc

    e.load_reference_trajectories() # loads the reference trajectory in olnmpc problem (for regularization)
    e.set_regularization_weights(R_w=0.0,Q_w=0.0,K_w=20.0) # R_w controls, Q_w states, K_w = control steps
    e.solve_olnmpc() # solves the olnmpc problem

    e.cycle_iterations()
    k += 1

    if  e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
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

plots = [('Y',()),('PO',()),('PO_fed',()),('W',()),('MY',()),('T_cw',()),('T',())]
#plots = [('PO',()),('T_cw',()),('T',())]
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

# print uncertatinty realization
###############################################################################
###         Plotting 1st Order Approximation of Confidence Region 
###############################################################################
try:
    error_bounds = {}
    estimates = [e.nmpc_trajectory[i,'e_pars'] for i in range(2,nfe)]
    plt.figure(l)
    dimension = 3 # dimension n of the n x n matrix = #DoF
    rhs_confidence = chi2.isf(1.0-0.95,dimension) # 0.1**2*5% measurment noise, 95% confidence level, dimension degrees of freedo
    rows = {}
    for r in range(1,24):
        A_dict = e.mhe_confidence_ellipsoids[r]
        center = [0,0]
        for m in range(dimension):
            rows[m] = np.array([A_dict[(m,i)] for i in range(dimension)])
        A = 1/rhs_confidence*np.array([np.array(rows[i]) for i in range(dimension)])
        center = np.array([0]*dimension)
        U, s, V = linalg.svd(A) # singular value decomposition 
        radii = 1/np.sqrt(s) # length of half axes, V rotation
        
        # transform in polar coordinates for simpler waz of plotting
#        theta = np.linspace(0.0, 2.0 * np.pi, 100) # angle = idenpendent variable
#        x = radii[0] * np.sin(theta) # x-coordinate
#        y = radii[1] * np.cos(theta) # y-coordinate
#        for i in range(len(x)):
#            [x[i],y[i]] = np.dot([x[i],y[i]], V) + center
#        plt.plot(x,y, label = str(r))

        # plot half axis
#    for p in range(dimension):
#        x = radii[p]*U[p][0]
#        y = radii[p]*U[p][1]
#        plt.plot([0,x],[0,y],color='red')
#    plt.xlabel(r'$\Delta A_i$')
#    plt.ylabel(r'$\Delta A_p$')
    
        
        dev = {(p,key):1e20 for p in e.p_noisy for key in e.p_noisy[p]}
        for p in e.p_noisy:
            p_mhe = getattr(e.lsmhe,p)
            for key in e.p_noisy[p]:
                pkey = key if key != () else None
                n = e.PI_indices[p,key]
                A_k = deepcopy(A)
                A_k[:,n] = 0.0
                A_k[n,:] = 0.0
                A_k[n,n] = 1.0
                Z_k = np.zeros(dimension)
                Z_k[:] = -A[:,n]
                Z_k[n] = 1.0
                a_k = A[n,:]
                D_k = np.linalg.solve(A_k,Z_k)
                dev_k = np.sqrt(1/np.dot(a_k,D_k))/p_mhe[pkey].value
                # use new interval if robustness_threshold < dev_k < confidence_threshold
                dev[(p,key)] = dev_k
        error_bounds[r] = deepcopy(dev)
            
    l +=1
    est_Ap = [estimates[i][('A',('p',))] for i in range(22)]
    est_Ai = [estimates[i][('A',('i',))] for i in range(22)]
    est_kA = [estimates[i][('kA',())] for i in range(22)]
    lb_Ap = [error_bounds[i][('A',('p',))] for i in range(2,24)]
    lb_Ai = [error_bounds[i][('A',('i',))] for i in range(2,24)]
    lb_kA = [error_bounds[i][('kA',())] for i in range(2,24)]
    err_Ap = [est_Ap[i]*lb_Ap[i] for i in range(22)]
    err_Ai = [est_Ai[i]*lb_Ai[i] for i in range(22)]
    err_kA = [est_kA[i]*lb_kA[i] for i in range(22)]
    x = [i for i in range(2,24)]
    plt.figure(l)
    plt.errorbar(x,est_Ap, yerr=err_Ap, fmt='bo', capsize=5)
    plt.plot([2,23],[e.plant_simulation_model.A['p'].value]*2,color='red',linestyle='dashed')
    plt.ylabel('A_p')
    plt.xlabel('iteration')
    plt.figure(l).savefig('Ap.pdf')
    l+=1
    plt.figure(l)
    plt.errorbar(x,est_Ai, yerr=err_Ai, fmt='bo', capsize=5)
    plt.plot([2,23],[e.plant_simulation_model.A['i'].value]*2,color='red',linestyle='dashed')
    plt.ylabel('A_i')
    plt.xlabel('iteration')
    plt.figure(l).savefig('Ai.pdf')
    l+=1
    plt.figure(l)
    plt.errorbar(x,est_kA, yerr=err_kA, fmt='bo', capsize=5)
    plt.plot([2,23],[e.plant_simulation_model.kA.value]*2,color='red',linestyle='dashed')
    plt.ylabel('kA')
    plt.xlabel('iteration')
    plt.figure(l).savefig('kA.pdf')
except KeyError:
    pass


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
plt.plot([0,max_tf],[443.15/100,443.15/100], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('Tad [K]')
    
l += 1
plt.figure(l)
for i in heat_removal:
    plt.plot(t[i],heat_removal[i], color='grey')
plt.plot([0,max_tf],[(273.15+150.0)/100.0,(273.15+150.0)/100.0], color='red', linestyle='dashed')
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

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:25:28 2017

@author: flemmingholtorf
"""
from __future__ import print_function
from pyomo.environ import *
from main.dync.MHEGen_multistage import MheGen
from main.mods.final_pwa.mod_class_cj_pwa_multistage import *
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

def run():
    states = ["PO","MX","MY","Y","W","PO_fed","T","T_cw"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
    x_noisy = ["PO","MX","MY","Y","W","T"] # all the states are noisy  
    x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)],"T":[()],"T_cw":[()]}
    p_noisy = {"A":[('p',),('i',)],'kA':[()]}
    u = ["u1", "u2"]
    u_bounds = {"u1": (-5.0, 5.0), "u2": (0.0, 3.0)} 
  
#    y = {"Y","PO", "W", "MY", "MX", "MW","m_tot",'T'}
#    y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()],'T':[()]}
    y = {"MY","Y","PO",'T'}#"m_tot"
    y_vars = {"MY":[()],"Y":[()],"PO":[()],'T':[()]} #"m_tot":[()],,"MW":[()]}
    nfe = 24
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
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),True,{('A',('p',)):1.0,('A',('i',)):1.0,('kA',()):1.0}) 
                elif s%s_max == 2:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
                elif s%s_max == 3:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
                elif s%s_max == 4:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
                elif s%s_max == 5:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
                elif s%s_max == 6:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
                elif s%s_max == 7:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
                elif s%s_max == 8:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
                else:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
        else:
            for s in range(1,s_max**nr+1):
                st[(i,s)] = (i-1,s,True,st[(i-1,s)][3])
    
    sr = s_max**nr
    
    e = MheGen(d_mod=SemiBatchPolymerization_multistage,
               d_mod_mhe = SemiBatchPolymerization,
               y=y,
               y_vars=y_vars,
               x_noisy=x_noisy,
               x_vars=x_vars,
               p_noisy=p_noisy,
               states=states,
               u=u,
               u_bounds = u_bounds,
               tf_bounds = tf_bounds,
               scenario_tree = st,
               robust_horizon = nr,
               s_max = sr,
               noisy_inputs = False,
               noisy_params = True,
               adapt_params = True,
               update_scenario_tree = False,
               process_noise_model = None,#'params_bias',
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
    e.create_nmpc()

    k = 1 
    for i in range(1,nfe):
        print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
        e.create_mhe()
        if i == 1:
            e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call = True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
            e.set_measurement_prediction(e.store_results(e.recipe_optimization_model))
            e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
            e.cycle_mhe(e.store_results(e.recipe_optimization_model),mcov,qcov,ucov,p_cov=pcov,first_call=True) 
            e.cycle_nmpc(e.store_results(e.recipe_optimization_model))
        else:
            e.plant_simulation(e.store_results(e.olnmpc),disturbance_src="parameter_noise",parameter_disturbance=v_param)
            e.set_measurement_prediction(e.store_results(e.forward_simulation_model))
            e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
            e.cycle_mhe(previous_mhe,mcov,qcov,ucov,p_cov=pcov) 
            e.cycle_nmpc(e.store_results(e.olnmpc))  
        
        # here measurement becomes available
        previous_mhe = e.solve_mhe(fix_noise=True) # solves the mhe problem
        #e.compute_confidence_ellipsoid()
        
        # solve the advanced step problems
        e.cycle_ics_mhe(nmpc_as=False,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc
        
        e.load_reference_trajectories()
        e.set_regularization_weights(K_w = 0.0, Q_w = 0.0, R_w = 0.0)
        e.solve_olnmpc() # solves the olnmpc problem
        
        e.cycle_iterations()
        k += 1
       
        #troubleshooting
        if  e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
            e.nmpc_trajectory[i,'solstat_mhe'] != ['ok','optimal'] or \
            e.plant_trajectory[i,'solstat'] != ['ok','optimal']:
            break
    # simulate the last step too
    
    for i in range(1,k):
        print('iteration: %i' % i)
        print('open-loop optimal control: ', end='')
        print(e.nmpc_trajectory[i,'solstat'],e.nmpc_trajectory[i,'obj_value'])
        print('constraint inf: ', e.nmpc_trajectory[i,'eps'])
        print('plant: ',end='')
        print(e.plant_trajectory[i,'solstat'])
    
    #print(e.st)
        
    e.plant_simulation(e.store_results(e.olnmpc))
    uncertainty_realization = {}
    for p in p_noisy:
        pvar_r = getattr(e.plant_simulation_model, p)
        pvar_m = getattr(e.recipe_optimization_model, p)
        for key in p_noisy[p]:
            pkey = None if key ==() else key
            print('delta_p ',p,key,': ',(pvar_r[pkey].value-pvar_m[pkey].value)/pvar_m[pkey].value)
            uncertainty_realization[(p,key)] = pvar_r[pkey].value
            
    tf = e.nmpc_trajectory[k, 'tf']
    if k == 24 and e.plant_trajectory[24,'solstat'] == ['ok','optimal']:
        return tf, e.plant_simulation_model.check_feasibility(display=True), e.pc_trajectory, uncertainty_realization
    else:
        return 'error', {'epc_PO_ptg': 'error', 'epc_mw': 'error', 'epc_unsat': 'error'}, 'error', 'error'

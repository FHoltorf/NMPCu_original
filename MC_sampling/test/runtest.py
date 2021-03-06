#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tu Dec 26 21:51:51 2017

@author: flemmingholtorf
"""
#### 

from __future__ import print_function
from pyomo.environ import *
from scipy.stats import chi2
from copy import deepcopy
from main.dync.MHEGen_adjusted import MheGen
from main.mods.test.mod_class_cj_pwa import *
from main.noise_characteristics_cj import * 
import itertools, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import time
import resource

def run(**kwargs):
    # monitor CPU-time
    CPU_t = {}
    
    scenario = kwargs.pop('scenario', {})
    
    states = ["PO","MX","MY","Y","W","PO_fed","T","T_cw"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
    x_noisy = ["PO","MX","MY","Y","W","T"] # all the states are noisy  
    x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)], "T":[()], "T_cw":[()]}
    p_noisy = {"A":[('p',),('i',)],'kA':[()]}#,'Hrxn_aux':[('p',)]}
    u = ["u1", "u2"]
    u_bounds = {"u1": (-5.0, 5.0), "u2": (0.0, 3.0)} 
    
    #y = {"Y","PO", "W", "MY", "MX", "MW","m_tot",'T'}
    #y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()],'T':[()]}
    y = {"MY","Y","PO",'T'}#"m_tot"
    y_vars = {"MY":[()],"Y":[()],"PO":[()],'T':[()]} #"m_tot":[()],,"MW":[()]}
    
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
               noisy_params = False,
               adapt_params = False,
               process_noise_model = 'params_bias',
               u_bounds=u_bounds,
               tf_bounds = tf_bounds,
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
    e.create_nmpc() # with tracking-type regularization
    #e.load_reference_trajectories()
    
    k = 1
    for i in range(1,nfe):
        print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
        e.create_mhe()
        if i == 1:
            #e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call = True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
            e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call = True, disturbance_src="parameter_scenario",scenario=scenario)
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
        t0 = time.time()
        t0_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        previous_mhe = e.solve_mhe(fix_noise=True) # solves the mhe problem
        tf_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        CPU_t[i,'mhe'] = time.time() - t0
        CPU_t[i,'mhe','cpu'] = tf_cpu.ru_utime - t0_cpu.ru_utime
        
        # solve the advanced step problems
        e.cycle_ics_mhe(nmpc_as=False,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc
    
        #e.load_reference_trajectories() # loads the reference trajectory in olnmpc problem (for regularization)
        e.set_regularization_weights(R_w=0.0,Q_w=0.0,K_w=1.0) # R_w controls, Q_w states, K_w = control steps
        t0 = time.time()
        t0_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        e.solve_olnmpc() # solves the olnmpc problem
        tf_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        CPU_t[i,'ocp'] = time.time() - t0
        CPU_t[i,'ocp','cpu'] = tf_cpu.ru_utime - t0_cpu.ru_utime
        
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
        print('lsmhe: ', end='')
        print(e.nmpc_trajectory[i,'solstat_mhe'],e.nmpc_trajectory[i,'obj_value_mhe'])
    
    e.plant_simulation(e.store_results(e.olnmpc),disturbance_src = "parameter_scenario",scenario=scenario)
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
        return tf, e.plant_simulation_model.check_feasibility(display=True), e.pc_trajectory, uncertainty_realization, CPU_t
    else:
        print(uncertainty_realization)
        sys.exit()
        return 'error', {'epc_PO_ptg': 'error', 'epc_mw': 'error', 'epc_unsat': 'error'}, 'error', uncertainty_realization, CPU_t

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:25:28 2017

@author: flemmingholtorf
"""
from __future__ import print_function
from main.dync.MHEGen import msMHEGen
from main.mods.SemiBatchPolymerization_cj.mod_class_stgen_cj import SemiBatchPolymerization_multistage
from main.mods.SemiBatchPolymerization_cj.mod_class_cj import SemiBatchPolymerization
from main.examples.SemiBatchPolymerization_cj.noise_characteristics_cj import *
import time, resource, sys
import numpy as np

def run(**kwargs):
    # monitor computational performance
    CPU_t = {}    
    scenario = kwargs.pop('scenario', {})    
    x_noisy = ["PO","MX","MY","Y","W","T"] # all the states are noisy  
    x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)],"T":[()],"T_cw":[()]}
    p_noisy = {"A":[('p',),('i',)],'kA':[()]}
    u = ["u1", "u2"]
    u_bounds = {"u1": (-5.0, 5.0), "u2": (0, 3.0)}
 
    cons = ['PO_ptg','unsat','mw','mw_ub','temp_b','T_max']
    y_vars = {"MY":[()],"Y":[()],"PO":[()],'T':[()]}
    
    noisy_ics = {'PO_ic':[()],'T_ic':[()],'MY_ic':[()],'MX_ic':[(1,)]}
    p_bounds = {('A', ('i',)):(-0.2,0.2),('A', ('p',)):(-0.2,0.2),('kA',()):(-0.2,0.2),
                ('PO_ic',()):(-0.02,0.02),('T_ic',()):(-0.005,0.005),
                ('MY_ic',()):(-0.01,0.01),('MX_ic',(1,)):(-0.003,0.003)}
    
    nfe = 24
    ncp = 3
    tf_bounds = [10.0*24.0/nfe, 30.0*24.0/nfe]
    
    pc = ['Tad','T']
    
    # scenario_tree
    st = {} # scenario tree : {parent_node, scenario_number on current stage, base node (True/False), scenario values {'name',(index):value}}
    s_max = 3
    nr = 2
    alpha = 0.2
    dummy ={(1, 2): {('A', ('p',)): 1-alpha, ('kA', ()): 1-alpha, ('A', ('i',)): 1-alpha, ('T_ic', ()): 1+0.005, ('MY_ic', ()): 1+0.01, ('PO_ic', ()): 1+0.02, ('MX_ic',(1,)): 1 - 0.005}, 
             (1, 3): {('A', ('p',)): 1-alpha, ('kA', ()): 1+alpha, ('A', ('i',)): 1-alpha, ('T_ic', ()): 1-0.005, ('MY_ic', ()): 1-0.01, ('PO_ic', ()): 1+0.02, ('MX_ic',(1,)): 1 - 0.005}, 
             (2, 3): {('A', ('p',)): 1-alpha, ('A', ('i',)): 1-alpha, ('kA', ()): 1-alpha},
             (2, 2): {('A', ('p',)): 1-alpha, ('A', ('i',)): 1-alpha, ('kA', ()): 1+alpha}}
    
    for i in range(1,nfe+1):
        if i < nr + 1:
            for s in range(1,s_max**i+1):
                if s%s_max == 1:
                    st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),True,{('A',('p',)):1.0,('A',('i',)):1.0,('kA',()):1.0}) 
                else:
                    scen = s%s_max if s%s_max != 0 else 3
                    st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,dummy[(i,scen)])
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
               update_scenario_tree = False,#True,
               uncertainty_set = p_bounds,
               process_noise_model = 'params_bias',
               confidence_threshold = alpha,
               robustness_threshold = 0.05,
               obj_type='economic',
               control_displacement_penalty=True,
               nfe_t=nfe,
               ncp_t=ncp,
               path_constraints=pc)
    
    ###############################################################################
    ###                                     NMPC
    ###############################################################################
    e.recipe_optimization()
    e.set_reference_state_trajectory(e.get_state_trajectory(e.recipe_optimization_model))
    e.set_reference_control_trajectory(e.get_control_trajectory(e.recipe_optimization_model))
    e.create_nmpc()    
    e.create_mhe()
    
    k = 1 
    for i in range(1,nfe):
        print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
        #e.plant_simulation(e.store_results(e.olnmpc),disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.plant_simulation(disturbance_src = "parameter_scenario",scenario=scenario)
        e.cycle_mhe() # only required for asMHE  
        t0 = time.time()
        t0_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        e.scenario_tree_generation(epc=cons[:4], pc=cons[4:],par_bounds=p_bounds,crit='con',noisy_ics=noisy_ics)
        tf_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        CPU_t[i,'stgen'] = time.time() - t0
        CPU_t[i,'stgen','cpu'] = tf_cpu.ru_utime - t0_cpu.ru_utime
        e.create_measurement(x_measurement,y_cov=mcov,q_cov=qcov,u_cov=ucov,p_cov=pcov)          
        e.cycle_nmpc()   
        # solve mhe problem
        t0 = time.time()
        t0_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        e.solve_mhe(fix_noise=True) # solves the mhe problem
        tf_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        CPU_t[i,'mhe'] = time.time() - t0
        CPU_t[i,'mhe','cpu'] = tf_cpu.ru_utime - t0_cpu.ru_utime
        if e.update_scenario_tree:  
            t0 = time.time()
            e.compute_confidence_ellipsoid()
            CPU_t[i,'cr'] = time.time() - t0
            CPU_t[i,'cr','cpu'] = tf_cpu.ru_utime - t0_cpu.ru_utime
        e.cycle_ics_mhe(nmpc_as=False,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc
        e.set_regularization_weights(K_w = 1.0, Q_w = 0.0, R_w = 0.0)
        t0 = time.time()
        t0_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        e.solve_olnmpc() # solves the olnmpc problem
        tf_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        CPU_t[i,'ocp'] = time.time() - t0
        CPU_t[i,'ocp','cpu'] = tf_cpu.ru_utime - t0_cpu.ru_utime
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
        print('lsmhe: ', end='')
        print(e.nmpc_trajectory[i,'solstat_mhe'],e.nmpc_trajectory[i,'obj_value_mhe'])
 
    #print(e.st)
        
    e.plant_simulation(disturbance_src = "parameter_scenario",scenario=scenario)
    uncertainty_realization = {}
    for p in p_noisy:
        pvar_r = getattr(e.plant_simulation_model, p)
        pvar_m = getattr(e.recipe_optimization_model, p)
        for key in p_noisy[p]:
            pkey = None if key ==() else key
            print('delta_p ',p,key,': ',(pvar_r[pkey].value-pvar_m[pkey].value)/pvar_m[pkey].value)
            uncertainty_realization[(p,key)] = pvar_r[pkey].value   
    tf = e.nmpc_trajectory[k,'tf']
    print('FINAL TIME ', tf)
    if k == 24 and e.plant_trajectory[24,'solstat'] == ['ok','optimal']:
        return tf, e.plant_simulation_model.check_feasibility(display=True), e.pc_trajectory, uncertainty_realization, CPU_t
    else:
        print(uncertainty_realization)
        return 'error', {'epc_PO_ptg': 'error', 'epc_mw': 'error', 'epc_unsat': 'error'}, 'error', uncertainty_realization, CPU_t

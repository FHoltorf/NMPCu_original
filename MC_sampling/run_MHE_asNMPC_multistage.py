#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:25:28 2017

@author: flemmingholtorf
"""
from __future__ import print_function
from pyomo.environ import *
# from nmpc_mhe.dync.MHEGen import MheGen
from main.dync.MHEGen_multistage import MheGen
from main.mods.mod_class import *
from main.mods.mod_class_multistage import *
import sys
import itertools, sys
import numpy as np
from main.noise_characteristics import *

# redirect system output to a file:
#sys.stdout = open('consol_output','w')

def run():
    states = ["PO","MX","MY","Y","W","PO_fed"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
    x_noisy = ["PO","MX","MY","Y","W","PO_fed"] # all the states are noisy  
    x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)]}
    #p_noisy = {"A":['p','i']}
    p_noisy = {"A":['p','i']}#,"Hrxn_aux":['p']}
    u = ["u1", "u2"]
    u_bounds = {"u1": (373.15/1e2, 443.15/1e2), "u2": (0, 3.0)} # 14.5645661157
    
    # measured variables
    y = ["heat_removal","m_tot","MW"] 
    y_vars = {"heat_removal":[()],"m_tot":[()],"MW":[()]}
    #y = {"PO", "Y", "W", "MY", "MX", "MW","m_tot"}
    #y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()]}
    
    
    pc = ['Tad','heat_removal']
    
    # scenario_tree
    st = {}
    s_max = 5
    nr = 1
    nfe = 24
    alpha = 0.2
    for i in range(1,nfe+1):
        if i < nr + 1:
            for s in range(1,s_max**i+1):
                if s%s_max == 1:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),True,{'p':1.0,'i':1.0})
                elif s%s_max == 2:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{'p':1.0+alpha,'i':1.0+alpha})
                elif s%s_max == 3:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{'p':1.0-alpha,'i':1.0+alpha})
                elif s%s_max == 4:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{'p':1.0+alpha,'i':1.0-alpha})
                else:
                    st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{'p':1.0-alpha,'i':1.0-alpha})
        else:
            for s in range(1,s_max**nr+1):
                st[(i,s)] = (i-1,s,True,st[(i-1,s)][3])
    
    sr = s_max**nr
    
    e = MheGen(d_mod=SemiBatchPolymerization_multistage,
               d_mod_mhe=SemiBatchPolymerization,
               y=y,
               x_noisy=x_noisy,
               y_vars=y_vars,
               x_vars=x_vars,
               p_noisy=p_noisy,
               states=states,
               u=u,
               scenario_tree = st,
               robust_horizon = nr,
               s_max = sr,
               noisy_inputs = False,
               noisy_params = True,
               adapt_params = True,
               update_scenario_tree = True,
               confidence_threshold = 0.2,
               robustness_threshold = 0.05,
               obj_type='economic',
               nfe_t=nfe,
               sens=None,
               diag_QR=True,
               del_ics=False,
               path_constraints=pc)
    e.recipe_optimization()
    e.set_reference_state_trajectory(e.get_state_trajectory(e.recipe_optimization_model))
    e.set_reference_control_trajectory(e.get_control_trajectory(e.recipe_optimization_model))
    e.generate_state_index_dictionary()
    e.create_enmpc() # with tracking-type regularization
    
    e.create_mhe()
    
    k = 1  
    for i in range(1,nfe):
        print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
        e.create_mhe()
        nfe_new = nfe - i
        if i == 1:
            e.plant_simulation(e.store_results(e.recipe_optimization_model),disturbances=v_disturbances,first_call=True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
            e.set_prediction(e.store_results(e.recipe_optimization_model))
            e.cycle_mhe(e.store_results(e.recipe_optimization_model),mcov,qcov,ucov, first_call=True) #adjusts the mhe problem according to new available measurements
            e.cycle_nmpc(e.store_results(e.recipe_optimization_model),nfe_new)
        else:
            e.plant_simulation(e.store_results(e.olnmpc),disturbances=v_disturbances,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
            e.set_prediction(e.store_results(e.forward_simulation_model))
            e.cycle_mhe(previous_mhe,mcov,qcov,ucov) 
            e.cycle_nmpc(e.store_results(e.olnmpc),nfe_new)   
    
        # solve the advanced step problems
        e.cycle_ics_mhe(nmpc_as=True,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc
    
        e.solve_olnmpc() # solves the olnmpc problem
    
        e.create_suffixes_nmpc()
        e.sens_k_aug_nmpc()
    
        e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
    
        # solve mhe problem
        e.solve_mhe(fix_noise=True) # solves the mhe problem
        previous_mhe = e.store_results(e.lsmhe)
        e.compute_confidence_ellipsoid()
    
        # update state estimate 
        e.update_state_mhe() # can compute offset within this function by setting as_nmpc_mhe_strategy = True
        # compute fast update for nmpc
        e.compute_offset_state(src_kind="estimated")
        e.sens_dot_nmpc()   
    
        # forward simulation for next iteration
        e.forward_simulation()
        e.cycle_iterations()
        k += 1
    
        #troubleshooting
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
    tf = e.nmpc_trajectory[k,'tf']
    if k == 24 and e.plant_trajectory[24,'solstat'] == ['ok','optimal']:
        return tf, e.plant_simulation_model.check_feasibility(display=True), e.pc_trajectory
    else:
        return 'error', {'epc_PO_ptg': 'error', 'epc_mw': 'error', 'epc_unsat': 'error'}, 'error'
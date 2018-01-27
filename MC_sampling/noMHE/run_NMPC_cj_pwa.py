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
from main.mods.mod_class_cj_pwa import *
from main.noMHE.noise_characteristics import * 
import itertools, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

def run():
    states = ["PO","MX","MY","Y","W","PO_fed","T","T_cw"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
    x_noisy = ["PO","MX","MY","Y","W","PO_fed","T"] # all the states are noisy  
    x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)], "T":[()], "T_cw":[()]}
    p_noisy = {"A":['p','i'],'kA':[()],'Hrxn_aux':['p']}
    u = ["u1", "u2"]
    u_bounds = {"u1": (-5.0, 5.0), "u2": (0.0, 3.0)} 
    
    nfe = 24
    tf_bounds = (10.0*24.0/nfe, 20.0*24.0/nfe)
    
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
               tf_bounds = tf_bounds,
               diag_QR=True,
               nfe_t=nfe,
               del_ics=False,
               sens=None,
               obj_type='economic',
               path_constraints=pc)
    
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

        e.cycle_ics() # writes the obtained initial conditions from mhe into olnmpc
        e.solve_olnmpc() # solves the olnmpc problem
        e.olnmpc.write_nl()
         
        e.cycle_iterations()
        k += 1
    
        if  e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
            e.plant_trajectory[i,'solstat'] != ['ok','optimal']:
            sys.exit()
    
    for i in range(1,k):
        print('iteration: %i' % i)
        print('open-loop optimal control: ', end='')
        print(e.nmpc_trajectory[i,'solstat'],e.nmpc_trajectory[i,'obj_value'])
        print('constraint inf: ', e.nmpc_trajectory[i,'eps'])
        print(e.plant_trajectory[i,'solstat'])
    
    e.plant_simulation(e.store_results(e.olnmpc))
    tf = e.nmpc_trajectory[k, 'tf']
    if k == 24 and e.plant_trajectory[24,'solstat'] == ['ok','optimal']:
        return tf, e.plant_simulation_model.check_feasibility(display=True), e.pc_trajectory
    else:
        return 'error', {'epc_PO_ptg': 'error', 'epc_mw': 'error', 'epc_unsat': 'error'}, 'error'
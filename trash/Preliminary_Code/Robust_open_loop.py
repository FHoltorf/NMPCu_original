#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:33:11 2017

@author: flemmingholtorf
@note: Merry Xmas
"""

from __future__ import print_function
from main.mods.mod_class_robust_optimal_control import *
import sys
import itertools, sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

__author__ = 'FHoltorf'

cons = ['PO_ptg','unsat','mw','temp_b','heat_removal_a']
p_noisy = {'A':['p','i'],'Hrxn':['p']}
iters = 0
iterlim = 100
converged = False
eps = 0.0
alpha = 0.2

n_p = 0
for key1 in p_noisy:
    for key2 in p_noisy[key1]:
        n_p += 1
Solver = SolverFactory('ipopt')
Solver.options["halt_on_ampl_error"] = "yes"
Solver.options["max_iter"] = 5000
Solver.options["tol"] = 1e-8
Solver.options["linear_solver"] = "ma57"
f = open("ipopt.opt", "w")
f.write("print_info_string yes")
f.close()
k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
k_aug.options["compute_dsdp"] = ""
#k_aug.options["no_barrier"] = ""
#k_aug.options["no_scale"] = ""

m = SemiBatchPolymerization(24,3) # nominal_model n_s=1
m.initialize_element_by_element()

# initialize
CPU_time = {}

backoff = {}
for i in cons:
    backoff_var = getattr(m,'xi_'+i)
    for index in backoff_var.index_set():
        try:
            backoff[('s_'+i,index)] = 0.0
            backoff_var[index].value = 0.0
        except KeyError:
            continue
ref = time.clock()
while (iters < iterlim and not(converged)):
    # solve optimization problem
    m.create_bounds()
    m.clear_aux_bounds()
    m.tf.setub(40)
    m.u1.unfix()
    m.u2.unfix()
    m.tf.unfix()
    for i in cons:
        slack = getattr(m, 's_'+i)
        for index in slack.index_set():
            slack[index].setlb(0)
    Solver.solve(m, tee=True)
    
    # solve square system
    m.u1.fix()
    m.u2.fix()
    m.tf.fix()
    m.clear_all_bounds()
    Solver.solve(m, tee=True)
    
    # compute sensitivities
    m.ipopt_zL_in.update(m.ipopt_zL_out)
    m.ipopt_zU_in.update(m.ipopt_zU_out)

    
    if iters == 0:
        m.var_order = Suffix(direction=Suffix.EXPORT)
        m.dcdp = Suffix(direction=Suffix.EXPORT)
        i = 1
        reverse_dict_pars = {}
        for p in p_noisy:
            for key in p_noisy[p]:
                dummy = 'dummy_constraint_p_' + p + '_' + key
                dummy_con = getattr(m, dummy)
                for index in dummy_con.index_set():
                    m.dcdp.set_value(dummy_con[index], i)
                    reverse_dict_pars[i] = (p,key)
                    i += 1
    
        i = 1
        reverse_dict_cons = {}
        for k in cons:
            s = getattr(m, 's_'+k)
            for index in s.index_set():
                if not(s[index].stale):
                    m.var_order.set_value(s[index], i)
                    reverse_dict_cons[i] = ('s_'+ k,index)
                    i += 1
            
    k_aug.solve(m, tee=True)
    
    sens = {}
    with open('dxdp_.dat') as f:
        reader = csv.reader(f, delimiter="\t")
        i = 1
        for row in reader:
            k = 1
            s = reverse_dict_cons[i]
            for col in row[1:]:
                p = reverse_dict_pars[k]
                sens[(s,p)] = float(col)
                k += 1
            i += 1
            
    
    # convergence check and update    
    converged = True
    for i in cons:
        backoff_var = getattr(m,'xi_'+i)
        for index in backoff_var.index_set():
            try:
                new_backoff = sum(abs(alpha*sens[(('s_'+i,index),reverse_dict_pars[k])]) for k in range(1,n_p+1))
                old_backoff = backoff[('s_'+i,index)]
                if backoff[('s_'+i,index)] - new_backoff < 0:
                    backoff[('s_'+i,index)] = new_backoff
                    backoff_var[index].value = new_backoff
                    if old_backoff - new_backoff < -eps:
                        converged = False
                else:
                    continue
            except KeyError:
                continue
    iters += 1
CPU_time['backoff'] = time.clock() - ref
##############################
    
print('#'*19 + '\n' + ' '*5 + ' converged ' + ' '*5 + '\n' + '#'*19)

nominal = SemiBatchPolymerization(24,3)
nominal.initialize_element_by_element()
ref = time.clock()
nominal.create_bounds()
nominal.clear_aux_bounds()
Solver.solve(nominal, tee=True)

CPU_time['nominal'] = time.clock() - ref
     
multimodel = SemiBatchPolymerization(24,3,n_s=27)
multimodel.initialize_element_by_element()

multimodel.p_A_par['p',1] = 1.0
multimodel.p_A_par['p',2] = 1.0 + alpha
multimodel.p_A_par['p',3] = 1.0 - alpha
multimodel.p_A_par['p',4] = 1.0
multimodel.p_A_par['p',5] = 1.0 + alpha
multimodel.p_A_par['p',6] = 1.0 - alpha
multimodel.p_A_par['p',7] = 1.0
multimodel.p_A_par['p',8] = 1.0 + alpha
multimodel.p_A_par['p',9] = 1.0 - alpha
multimodel.p_A_par['p',10] = 1.0
multimodel.p_A_par['p',11] = 1.0 + alpha
multimodel.p_A_par['p',12] = 1.0 - alpha
multimodel.p_A_par['p',13] = 1.0
multimodel.p_A_par['p',14] = 1.0 + alpha
multimodel.p_A_par['p',15] = 1.0 - alpha
multimodel.p_A_par['p',16] = 1.0
multimodel.p_A_par['p',17] = 1.0 + alpha
multimodel.p_A_par['p',18] = 1.0 - alpha
multimodel.p_A_par['p',19] = 1.0
multimodel.p_A_par['p',20] = 1.0 + alpha
multimodel.p_A_par['p',21] = 1.0 - alpha
multimodel.p_A_par['p',22] = 1.0
multimodel.p_A_par['p',23] = 1.0 + alpha
multimodel.p_A_par['p',24] = 1.0 - alpha
multimodel.p_A_par['p',25] = 1.0
multimodel.p_A_par['p',26] = 1.0 + alpha
multimodel.p_A_par['p',27] = 1.0 - alpha

multimodel.p_A_par['i',1] = 1.0 
multimodel.p_A_par['i',2] = 1.0 
multimodel.p_A_par['i',3] = 1.0 
multimodel.p_A_par['i',4] = 1.0 - alpha
multimodel.p_A_par['i',5] = 1.0 - alpha
multimodel.p_A_par['i',6] = 1.0 - alpha
multimodel.p_A_par['i',7] = 1.0 + alpha
multimodel.p_A_par['i',8] = 1.0 + alpha
multimodel.p_A_par['i',9] = 1.0 + alpha
multimodel.p_A_par['i',10] = 1.0 
multimodel.p_A_par['i',11] = 1.0 
multimodel.p_A_par['i',12] = 1.0 
multimodel.p_A_par['i',13] = 1.0 - alpha
multimodel.p_A_par['i',14] = 1.0 - alpha
multimodel.p_A_par['i',15] = 1.0 - alpha
multimodel.p_A_par['i',16] = 1.0 + alpha
multimodel.p_A_par['i',17] = 1.0 + alpha
multimodel.p_A_par['i',18] = 1.0 + alpha
multimodel.p_A_par['i',19] = 1.0
multimodel.p_A_par['i',20] = 1.0 
multimodel.p_A_par['i',21] = 1.0 
multimodel.p_A_par['i',22] = 1.0 - alpha
multimodel.p_A_par['i',23] = 1.0 - alpha
multimodel.p_A_par['i',24] = 1.0 - alpha
multimodel.p_A_par['i',25] = 1.0 + alpha
multimodel.p_A_par['i',26] = 1.0 + alpha
multimodel.p_A_par['i',27] = 1.0 + alpha

multimodel.p_Hrxn_par['p',1] = 1.0
multimodel.p_Hrxn_par['p',2] = 1.0 
multimodel.p_Hrxn_par['p',3] = 1.0
multimodel.p_Hrxn_par['p',4] = 1.0
multimodel.p_Hrxn_par['p',5] = 1.0
multimodel.p_Hrxn_par['p',6] = 1.0 
multimodel.p_Hrxn_par['p',7] = 1.0
multimodel.p_Hrxn_par['p',8] = 1.0 
multimodel.p_Hrxn_par['p',9] = 1.0
multimodel.p_Hrxn_par['p',10] = 1.0 + alpha
multimodel.p_Hrxn_par['p',11] = 1.0 + alpha
multimodel.p_Hrxn_par['p',12] = 1.0 + alpha
multimodel.p_Hrxn_par['p',13] = 1.0 + alpha
multimodel.p_Hrxn_par['p',14] = 1.0 + alpha
multimodel.p_Hrxn_par['p',15] = 1.0 + alpha
multimodel.p_Hrxn_par['p',16] = 1.0 + alpha
multimodel.p_Hrxn_par['p',17] = 1.0 + alpha
multimodel.p_Hrxn_par['p',18] = 1.0 + alpha
multimodel.p_Hrxn_par['p',19] = 1.0 - alpha
multimodel.p_Hrxn_par['p',20] = 1.0 - alpha
multimodel.p_Hrxn_par['p',21] = 1.0 - alpha
multimodel.p_Hrxn_par['p',22] = 1.0 - alpha
multimodel.p_Hrxn_par['p',23] = 1.0 - alpha
multimodel.p_Hrxn_par['p',24] = 1.0 - alpha
multimodel.p_Hrxn_par['p',25] = 1.0 - alpha
multimodel.p_Hrxn_par['p',26] = 1.0 - alpha
multimodel.p_Hrxn_par['p',27] = 1.0 - alpha

ref = time.clock()
multimodel.create_bounds()
multimodel.clear_aux_bounds()
multimodel.tf.setub(None)
Solver.solve(multimodel, tee=True)
CPU_time['multimodel'] = time.clock() - ref


# plot trajectories
multimodel.plot_profiles([multimodel.MY,multimodel.Y,multimodel.PO,multimodel.T,multimodel.F,multimodel.heat_removal, multimodel.Tad], label = 'multimodel')        
nominal.plot_profiles([nominal.MY,nominal.Y,nominal.PO,nominal.T,nominal.F,nominal.heat_removal,nominal.Tad],label ='nominal')
m.plot_profiles([m.MY,m.Y,m.PO,m.T,m.F,m.heat_removal,m.Tad],label='backoff')

# print timings
print(CPU_time)



    
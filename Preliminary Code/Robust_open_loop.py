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

end = ['PO_ptg','unsat','mw','temp_b','heat_removal_a']
dummies = ['dummy_constraint1','dummy_constraint2']
n_p = 2 
   
m = SemiBatchPolymerization(24,3) # nominal_model n_s=1
m.initialize_element_by_element()
iters = 0
iterlim = 100
converged = False
eps = 1e-1
alpha = 0.3

# initialize
CPU_time = {}
CPU_time['reference'] = time.clock()
backoff = {}
for i in end:
    backoff_var = getattr(m,'xi_'+i)
    for index in backoff_var.index_set():
        try:
            backoff[(('s_'+i,index),'p1')] = 0.0
            backoff_var[index].value = 0.0
        except KeyError:
            continue

while (iters < iterlim and not(converged)):
    # solve optimization problem
    m.create_bounds()
    m.clear_aux_bounds()
    m.tf.setub(None)
    m.u1.unfix()
    m.u2.unfix()
    m.tf.unfix()
    for i in end:
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
        for dummy in dummies:
            dummy_con = getattr(m, dummy)
            for index in dummy_con.index_set():
                m.dcdp.set_value(dummy_con[index], i)
                i += 1
    
        i = 1
        reverse_dict = {}
        for k in end:
            s = getattr(m, 's_'+k)
            for index in s.index_set():
                if not(s[index].stale):
                    m.var_order.set_value(s[index], i)
                    reverse_dict[i] = ('s_'+ k,index)
                    i += 1
            
    k_aug.solve(m, tee=True)
    
    sens = {}
    with open('dxdp_.dat') as f:
        reader = csv.reader(f, delimiter="\t")
        i = 1
        for row in reader:
            k = 1
            s = reverse_dict[i]
            for col in row[1:]:
                sens[(s,'p'+str(k))] = float(col)
                k += 1
            i += 1
    # convergence check and update    
    converged = True
    for i in end:
        backoff_var = getattr(m,'xi_'+i)
        for index in backoff_var.index_set():
            try:
                new_backoff = sum(abs(alpha*sens[(('s_'+i,index),'p'+str(k))]) for k in range(1,n_p+1))
                old_backoff = backoff[(('s_'+i,index),'p1')]
                if backoff[(('s_'+i,index),'p1')] - new_backoff < 0:
                    backoff[(('s_'+i,index),'p1')] = new_backoff
                    backoff_var[index].value = new_backoff
                    if old_backoff - new_backoff < -eps:
                        converged = False
                else:
                    continue
            except KeyError:
                continue
    iters += 1
CPU_time['backoff'] = time.clock() - sum(CPU_time[k] for k in CPU_time)
##############################
    
print('#'*19 + '\n' + ' '*5 + ' converged ' + ' '*5 + '\n' + '#'*19)

nominal = SemiBatchPolymerization(24,3)
nominal.initialize_element_by_element()
nominal.create_bounds()
nominal.clear_aux_bounds()
Solver.solve(nominal, tee=True)

CPU_time['nominal'] = time.clock() - sum(CPU_time[k] for k in CPU_time)
     
multimodel = SemiBatchPolymerization(24,3,n_s=9)
multimodel.initialize_element_by_element()

multimodel.p3[1] = 1.0
multimodel.p3[2] = 1.0 + alpha
multimodel.p3[3] = 1.0 - alpha
multimodel.p3[4] = 1.0
multimodel.p3[5] = 1.0 + alpha
multimodel.p3[6] = 1.0 - alpha
multimodel.p3[7] = 1.0
multimodel.p3[8] = 1.0 + alpha
multimodel.p3[9] = 1.0 - alpha

multimodel.p4[1] = 1.0
multimodel.p4[2] = 1.0 
multimodel.p4[3] = 1.0 
multimodel.p4[4] = 1.0 + alpha
multimodel.p4[5] = 1.0 + alpha
multimodel.p4[6] = 1.0 + alpha
multimodel.p4[7] = 1.0 - alpha
multimodel.p4[8] = 1.0 - alpha
multimodel.p4[9] = 1.0 - alpha


multimodel.create_bounds()
multimodel.clear_aux_bounds()
multimodel.tf.setub(None)
Solver.solve(multimodel, tee=True)
CPU_time['multimodel'] = time.clock() - sum(CPU_time[k] for k in CPU_time) 
multimodel.plot_profiles([multimodel.MY,multimodel.Y,multimodel.PO,multimodel.T,multimodel.F,multimodel.heat_removal, multimodel.Tad], label = 'multimodel')        
nominal.plot_profiles([nominal.MY,nominal.Y,nominal.PO,nominal.T,nominal.F,nominal.heat_removal,nominal.Tad],label ='nominal')
m.plot_profiles([m.MY,m.Y,m.PO,m.T,m.F,m.heat_removal,m.Tad],label='backoff')



    
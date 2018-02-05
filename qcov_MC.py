#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:08:39 2018

@author: flemmingholtorf
"""

from __future__ import print_function
from mods.cj.mod_class_cj_pwa import *
#from mods.no_cj.mod_class_robust_optimal_control import *
from copy import deepcopy
from pyomo.core import *
from pyomo.environ import *
import numpy as np
import pickle
import sys,csv


# qcov = E(dw^T*dw)(t)

# nominal trajectory
e = SemiBatchPolymerization(24,3)
e.initialize_element_by_element()
e.create_output_relations()
e.create_bounds()
e.clear_aux_bounds()

ip = SolverFactory('ipopt')
ip.options["linear_solver"] = "ma57"

nom_res = e.save_results(ip.solve(e,tee=True))

x_noisy = {"T":[()],"PO":[()],"MX":[(0,),(1,)],"MY":[()],"Y":[()],"W":[()]}
p_noisy = {'A':['p','i'],
           'kA':[()]}
uncertainty_level = {('A',('p')):0.1,('A',('i')):0.1,('kA',()):0.1}
nominal_values = {('A',('p')):e.A['p'].value,('A',('i')):e.A['i'].value,('kA',()):e.kA.value}

# sampling
sample_size = 100
sampled_res = {}
m = SemiBatchPolymerization(24,3)
m.create_output_relations()
m.create_bounds()
m.clear_aux_bounds()
m.clear_all_bounds()
m.deactivate_epc()
m.deactivate_pc()
m.eobj.deactivate()
for index in m.k_l.index_set():
    m.k_l[index].setub(20.0)
   
k=0
for j in range(sample_size):
    print(j)
    for var in m.component_objects(Var):
        for index in var.index_set():
            var[index].value = nom_res[(var.name,index)]
    m.u1.fix()
    m.u2.fix()
    m.tf.fix()
    for p in p_noisy:
        par = getattr(m,p)
        for key in p_noisy[p]:
            if key != ():
                par[key].value = nominal_values[(p,key)] * (1 + np.random.normal(0,uncertainty_level[(p,key)]))
            else:
                par.value = nominal_values[(p,key)] * (1 + np.random.normal(0,uncertainty_level[(p,key)]))
    out = ip.solve(m,tee=False)    
    if [str(out.solver.status), str(out.solver.termination_condition)] == ['ok','optimal']:
        sampled_res[j] = m.save_results(out)
        k += 1
    else:
        sampled_res[j] = 'not converged'
sample_size=k
        
# compute covariance matrix
dw = {}
for j in range(sample_size):
    dw[j] = {}
    if sampled_res[j] != 'not converged':
        for x in x_noisy:
            for key in x_noisy[x]:
                for t in range(1,25):
                    dw[j][x,t] = sampled_res[j][(x,(t,3)+key)]-nom_res[(x,(t,3)+key)]
                
row_order = {0:("Y",()),1:("PO",()),2:("MY",()),3:("T",()),4:("MX",(0,)),5:("MX",(1,)),6:("W",())} 
nx = len(row_order)

_Q = {}
_Q_inv = {}
for t in range(1,25):
    _Q[t] = np.zeros((nx,nx))
    sum_Q = _Q[t]
    for j in range(sample_size):
        _dw = np.zeros((nx,1))
        for i in range(nx):
            _dw[i,0]=dw[j][row_order[i][0],t]
            sum_Q += np.dot(_dw,_dw.transpose())
    _Q[t] = 1.0/sample_size*sum_Q
    _Q_inv[t] = np.linalg.inv(_Q[t])

qcov = {}
# add unused states with 0.0 
for t in range(1,25):
    aux = {}
    for i in range(nx):
        for j in range(nx):
            aux[row_order[i],row_order[j]] = _Q_inv[t][i][j]
    qcov[t] = deepcopy(aux)    
#set 0
qcov[0] = {}
for i in range(nx):
    for j in range(nx):
        qcov[0][row_order[i],row_order[j]] = 0.0
f = open('qcov_cj.pckl','wb')
pickle.dump(qcov, f)
f.close()


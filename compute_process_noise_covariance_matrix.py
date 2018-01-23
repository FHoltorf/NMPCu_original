#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:55:44 2018

@author: flemmingholtorf
"""
from __future__ import print_function
from mods.mod_class_robust_optimal_control import *
import sys,csv
import numpy.linalg as linalg
from pyomo.core import *
from pyomo.environ import *
import numpy as np
import pickle
from copy import deepcopy

ip = SolverFactory('ipopt')
ip.options["linear_solver"] = "ma57"
m = SemiBatchPolymerization(24,3)
m.initialize_element_by_element()
m.create_bounds()
m.clear_aux_bounds()
ip.solve(m,tee=True)

m.u1.fix()
m.u2.fix()
m.clear_all_bounds()

k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
k_aug.options["compute_dsdp"] = ""

m.var_order = Suffix(direction=Suffix.EXPORT)
m.dcdp = Suffix(direction=Suffix.EXPORT)

i = 1
p_noisy = {'A':['p','i']}
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
states = {"PO":[()], "Y":[()]}#, "W":[()], "MY":[()], "MX":[(0,),(1,)]} #, "PO_fed":[()] removed since not subject to disturbances

reverse_dict_cons = {}
for x in states:
    xvar = getattr(m,x)
    for index in xvar.index_set():
        if not(xvar[index].stale) and index[1] == 3:
            m.var_order.set_value(xvar[index], i)
            reverse_dict_cons[i] = (x,index)
            i += 1
            
            
            
k_aug.solve(m,tee=True)
dfdp = {}
with open('dxdp_.dat') as f:
    reader = csv.reader(f, delimiter="\t")
    i = 1
    for row in reader:
        k = 1
        x = reverse_dict_cons[i]
        for col in row[1:]:
            p = reverse_dict_pars[k]
            pvar = getattr(m,p[0])
            dfdp[(x,p)] = float(col) ##*1/pvar[p[1]].value
            k += 1
        i += 1




# which states are useless?
liste = []
for key in dfdp:
    if abs(dfdp[key]) <= 1e-2:
        liste.append(key)


#########################
# matrix dfdp[t]
row_order = {0:("Y",()),1:("PO",())} #1:("Y",()), 4:("MX",(0,)), # 2:("W",()), #, 3:("MX",(1,)) # 0:("PO",()),
_dfdp = {}
n_x = len(row_order)
n_p = 2
for t in range(24):
    _dfdp[t] = np.zeros((n_x,n_p))
    for i in range(n_x):
        for j in range(n_p):
            p = reverse_dict_pars[j+1]
            if row_order[i][1] != ():
                x = (row_order[i][0],(t+1,3,row_order[i][1][0],1))
            else:
                x = (row_order[i][0],(t+1,3,1))
            _dfdp[t][i][j] = dfdp[(x,p)]
            
# covariance matrix of parameters
_Vp = np.array([[0.2**2, 0.0],[0.0,0.2**2]])
_Q = {}
_Q_inv = {}
for t in range(24):
    _Q[t] = np.dot(_dfdp[t],np.dot(_Vp,_dfdp[t].transpose()))
    if t > 0:
        _Q_inv[t] = np.linalg.inv(_Q[t])
        
qcov = {}
for t in range(24):
    aux = {}
    for i in range(n_x):
        for j in range(n_x):
            if t > 0:
                aux[row_order[i],row_order[j]] = _Q_inv[t][i][j]
    qcov[t] = deepcopy(aux)

f = open('qcov.pckl','wb')
pickle.dump(qcov, f)
f.close()

    




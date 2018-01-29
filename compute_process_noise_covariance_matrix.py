#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:55:44 2018

@author: flemmingholtorf
"""
from __future__ import print_function
from mods.cj.mod_class_cj_pwa_robust_optimal_control import *
#from mods.no_cj.mod_class_robust_optimal_control import *
from copy import deepcopy
from pyomo.core import *
from pyomo.environ import *
import numpy as np
import pickle
import sys,csv


ip = SolverFactory('ipopt')
ip.options["linear_solver"] = "ma57"
m = SemiBatchPolymerization(24,3)
m.initialize_element_by_element()
m.create_bounds()
m.clear_aux_bounds()
ip.solve(m,tee=True)

m.u1.fix()
m.u2.fix()
m.tf.fix()
m.eps_pc.fix()
m.eps.fix()
m.deactivate_epc()
m.deactivate_pc()
m.clear_all_bounds()

k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
k_aug.options["compute_dsdp"] = ""

m.var_order = Suffix(direction=Suffix.EXPORT)
m.dcdp = Suffix(direction=Suffix.EXPORT)

i = 1
p_noisy = {'A':['p','i'],'kA':[()]}
reverse_dict_pars = {}
for p in p_noisy:
    for key in p_noisy[p]:
        if key != ():  
            dummy = 'dummy_constraint_p_' + p + '_' + key
        else:
            dummy = 'dummy_constraint_p_' + p
        dummy_con = getattr(m, dummy)
        for index in dummy_con.index_set():
            m.dcdp.set_value(dummy_con[index], i)
            reverse_dict_pars[i] = (p,key)
            i += 1
            
i = 1
states = {"Y":[()],"PO":[()], "W":[()], "MY":[()],"MX":[(0,),(1,)],"PO_fed":[()],"T":[()]} #, "PO_fed":[()] removed since not subject to disturbances

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
    if abs(dfdp[key]) <= 1e-5:
        liste.append(key)

#########################
# matrix dfdp[t]
row_order = {0:("Y",()),1:("PO",()),2:("T",())}#,4:("T",())} #1:("PO",()),
# 2:("W",()),3:("MX",(0,)),3:("MX",(1,)),2:("MY",()), ,4:("T",()),2:("MX",(1,)),
_dfdp = {}
n_x = len(row_order)
n_p = 3
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
_Vp = np.diag([0.2**2,0.2**2,0.2**2])
_Q = {}
_Q_inv = {}
for t in range(24):
    _Q[t] = np.dot(_dfdp[t],np.dot(_Vp,_dfdp[t].transpose()))
    #if t > 0:
    _Q_inv[t] = np.linalg.inv(_Q[t])
        
qcov = {}
for t in range(24):
    aux = {}
    for i in range(n_x):
        for j in range(n_x):
            aux[row_order[i],row_order[j]] = _Q_inv[t][i][j]
    qcov[t] = deepcopy(aux)

f = open('qcov_cj.pckl','wb')
pickle.dump(qcov, f)
f.close()

    




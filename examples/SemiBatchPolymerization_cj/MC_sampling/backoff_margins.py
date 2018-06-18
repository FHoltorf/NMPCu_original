#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:12:04 2018

@author: flemmingholtorf
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt 

path = 'results/125grid/timeinvariant/parest/nominal/'
T_ub = (273.15+150)/1e2
T_ad_ub = (273.15+170)/1e2
f = open(path + 'epc.pckl','rb')
endpoint_constraints = pickle.load(f)
f.close()
  
f = open(path + 'final_times.pckl','rb')
tf = pickle.load(f)
f.close()
  
f = open(path + 'path_constraints.pckl','rb')
path_constraints = pickle.load(f)
f.close()

iters = len(tf)

constraint_name = []
for constraint in endpoint_constraints[0]:
    if constraint == 'feasible':
        continue
    constraint_name.append(constraint)

names = ['Tad','T']
setpoint = {'Tad':T_ad_ub,'T':T_ub}
y = {}
backoff = {}
tv_backoff = {}
for name in names:
# timevariant backoffs
    for fe in range(1,25):
        for cp in range(1,4):
            y[name,fe] = [path_constraints[i][(name,(fe,(cp,)))] for i in range(iters)]#for fe in range(1,25) for cp in range(1,4)
            std = np.std(y[name,fe]) 
            mu = np.mean(y[name,fe])
            n = 1000
            y[name,fe] = [i for i in y[name,fe] if i >= mu-n*std and i <= mu+n*std]
            y[name,fe].sort()
            tv_backoff[name,fe] = max(0,y[name,fe][-1] - setpoint[name])*1e2
            tv_backoff[name+'_min',fe] = -1*min(0,y[name,fe][0] - 3.7315)*1e2
    backoff[name,'overall'] = max([tv_backoff[name,fe] for fe in range(1,25)])
    backoff[name+'_min','overall'] = max([tv_backoff[name+'_min',fe] for fe in range(1,25)])
    
names = ['epc_PO_ptg','epc_mw','epc_unsat']
setpoint = {'epc_PO_ptg':120,'epc_mw':-949.5,'epc_unsat':0.032}
for name in names:
    y[name]=[endpoint_constraints[i][name] for i in range(iters)]
    std = np.std(y[name]) 
    mu = np.mean(y[name])
    n = 1e10
    y[name] = [i for i in y[name] if i >= mu-n*std and i <= mu+n*std]
    y[name].sort()
    backoff[name] = -y[name][0] #+ setpoint[name]
print(backoff)

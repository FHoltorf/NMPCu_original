#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:39:04 2018

@author: flemmingholtorf
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt # for 2d plots
from mpl_toolkits.mplot3d import Axes3D # for 3d histogram plots
import sys

#folders = ['online_estimation','multistage','backoff','standard']
p1 = 'finalfinal/timeinvariant/parest/'
p2 = 'finalfinal/timeinvariant/standard/'
folders = [p1+'nominal',p1+'SBBM',p1+'multistage',p1+'multistage_stgen']#p1+'nominal_bo',
directory = 'results/finalfinal/' # save overall plots here
method = {p1+'nominal':'nominal',
          #p1+'nominal_bo':'NMPC-bo',
          p1+'SBBM':'SBBM',
          p1+'multistage':'ms',
          p1+'multistage_stgen':'ms-SBSG'}
scaling = {'epc_mw':1e0, 'epc_PO_ptg':1e1,'epc_unsat':1e-3}
comparison = {}
for folder in folders: 
    print(folder)
    path = 'results/'+folder+'/' 
    color = ['green','yellow','blue']
    
    f = open(path + 'epc.pckl','rb')
    endpoint_constraints = pickle.load(f)
    f.close()
    
    f = open(path + 'final_times.pckl','rb')
    tf = pickle.load(f)
    f.close()
    
    f = open(path + 'path_constraints.pckl','rb')
    path_constraints = pickle.load(f)
    f.close()
    
    f = open(path + 'runtime.pckl','rb')
    runtime = pickle.load(f)
    f.close()
    
    f = open(path + 'CPU_t.pckl','rb')
    CPU_t = pickle.load(f)
    f.close()
    
    f = open(path + 'scenarios','rb')
    scenarios = pickle.load(f)
    f.close()
    
    f = open(path + 'uncertainty_realization.pckl','rb')
    uncertainty_realization = pickle.load(f)
    f.close()
    
    constraint_name = []
    #endpoint constraints
    iters = len(tf)
    for constraint in endpoint_constraints[0]:
        if constraint == 'feasible':
            continue
        constraint_name.append(constraint)
    
    for k in range(3):
        color[k]
        x = [endpoint_constraints[i][constraint_name[k]] for i in range(iters) if endpoint_constraints[i][constraint_name[k]] != 'error']
        # compute standard deviation
        std = np.std(x) 
        mu = np.mean(x)
        n = 100
        # remove outliers (not in interval +-n x std)
        x = [i for i in x if i >= mu-n*std and i <= mu+n*std]
        comparison[folder,constraint_name[k]] = max(0,-min(x))/scaling[constraint_name[k]]
      
    # path constraints
    Tad_max = -1e8
    T_max = -1e8
    T_min = 1e8
    for i in path_constraints: # loop over all runs
        if path_constraints[i] =='error':
            continue
        #heat_removal[i] = []
        for fe in range(1,25):
            for cp in range(1,4):        
                #heat_removal[i].append(path_constraints[i]['heat_removal',(fe,(cp,))]*92048.0/60.0)
                if T_max < path_constraints[i]['T',(fe,(cp,))]*100.0:
                    T_max = path_constraints[i]['T',(fe,(cp,))]*100.0
                if Tad_max < path_constraints[i]['Tad',(fe,(cp,))]*100.0:
                    Tad_max = path_constraints[i]['Tad',(fe,(cp,))]*100.0
                if T_min > path_constraints[i]['T',(fe,(cp,))]*100.0:
                    T_min = path_constraints[i]['T',(fe,(cp,))]*100.0

    max_tf = max([tf[i] for i in tf if endpoint_constraints[i]['feasible'] != 'crashed'])   
    avg_tf = sum(tf[i] for i in tf if endpoint_constraints[i]['feasible'] != 'crashed')/sum(1 for i in tf if endpoint_constraints[i]['feasible'] != 'crashed')
    min_tf = min([tf[i] for i in tf if endpoint_constraints[i]['feasible'] != 'crashed'])
    
    comparison[folder,'T_max'] = max(0,T_max - 423.15)
    comparison[folder,'Tad_max'] = max(0,Tad_max - 443.15)
    comparison[folder,'T_min'] = max(0,-T_min + 373.15)
    comparison[folder,'max_tf'] = max_tf 
    comparison[folder,'avg_tf'] = avg_tf
    comparison[folder,'min_tf'] = min_tf

fig, ax1 = plt.subplots()
ind = np.arange(len(folders))
width = 1.0/6.0 - 0.05*0.0
bars = {}
xticks = [method[f] for f in folders]
xspacing = ind + 0.5
i = 0.0

labels = {'T_max':r'$T^{max} \, [K]$','Tad_max':r'$T^{max}_{ad} \, [K]$','T_min':r'$T^{min} \, [K]$','epc_mw':r'$NAMW \, [\frac{g}{mol}]$', 'epc_PO_ptg':r'$unreac \, [10^{2} \cdot PPM]$','epc_unsat':r'$unsat \, [10^{3} \cdot \frac{mol}{g_{PO}}]$','max_tf':r'$t_f^{max}$','avg_tf':r'$t_f^{avg}$','min_tf':r'$t_f^{min}$'}
for con in constraint_name:
    vals = []
    for folder in folders:
        vals.append(comparison[folder,con])
    vals = np.array(vals)#/max(abs(min(vals)),max(vals))*100
    bars[con] = ax1.bar(ind+width*i, vals,width,label=labels[con])
    i += 1
for con in ['T_max','Tad_max','T_min']:
    vals = []
    for folder in folders:
        vals.append(comparison[folder,con])
    vals = np.array(vals)#/max(abs(min(vals)),max(vals))*100
    bars[con] = ax1.bar(ind+width*i, vals, width, label=labels[con])
    i += 1
ax1.set_xticks(xspacing)
ax1.set_xticklabels(xticks) #can hold text
ax1.set_ylabel('maximum constraint violation')
ax1.legend(loc ='lower center', ncol=3, fancybox=True, bbox_to_anchor=(0.5,-0.3))
ax2 = ax1.twinx()
color = ['r','k','g']
k = 0
for t in ['max_tf','avg_tf','min_tf']:
    vals = []
    for folder in folders:
        vals.append(comparison[folder,t])
    ax2.plot(xspacing,vals,color[k]+'-o',label=labels[t])
    k += 1
ax2.set_ylim([0,750])
ax2.set_ylabel('final batch time [min]')
ax2.legend(loc='upper right')
fig.tight_layout()
plt.show()
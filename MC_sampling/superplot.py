#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:39:04 2018

@author: flemmingholtorf
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt # for 2d plots
from mpl_toolkits.mplot3d import Axes3D # for 3d histogram plots
import sys

# use pgf backend


#folders = ['online_estimation','multistage','backoff','standard']
#p1 = 'final/timeinvariant/parest/'
#p2 = 'final/timeinvariant/standard/'
#folders = [p1+'nominal',p1+'nominal_bo',p1+'multistage',p1+'multistage_stgen',p1+'SBBM']
#directory = 'results/' # save overall plots here
#method = {p1+'nominal':'nominal',
#          p1+'nominal_bo':'NMPC-bo',
#          p1+'SBBM':'NMPC-SBBM',
#          p1+'multistage':'msNMPC',
#          p1+'multistage_stgen':'msNMPC-SBSG'}
# p2 = 'finalfinal/timeinvariant/standard/'
p1 = '125grid/timeinvariant/parest/'
folders = [p1+'nominal',p1+'ms',p1+'SBSG',
           p1+'SBBM',
           p1+'nominal_bo']#,p1+'SBSG1stage'
directory = 'results/125grid/' # save overall plots here
method = {p1+'nominal':'NMPC',
          p1+'ms':'msNMPC',
          p1+'nominal_bo':'NMPC-BO',
         # p1+'SBSG1stage':'SBSG1stage',
          p1+'SBSG':'msNMPC-SBSG',
          p1+'SBBM':'NMPC-SBBM'}
scaling = {'epc_mw':1e0, 'epc_PO_ptg':1e2,'epc_unsat':1e-3}
comparison = {}
# Baseline
f = open('results/125grid/baseline/lower_bound.pckl','rb')
lb = pickle.load(f)
f.close()

comparison['baseline','max_tf'] = max([lb[i] for i in lb])
comparison['baseline','min_tf'] = min([lb[i] for i in lb])
comparison['baseline','avg_tf'] = sum([lb[i] for i in lb])/len(lb)
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
    
    f = open(path + 'scenarios.pckl','rb')
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
        x = [endpoint_constraints[i][constraint_name[k]] for i in range(iters) if endpoint_constraints[i][constraint_name[k]] != 'error']
        # compute standard deviation
        std = np.std(x) 
        mu = np.mean(x)
        n = 100
        # remove outliers (not in interval +-n x std)
        x = [i for i in x if i >= mu-n*std and i <= mu+n*std]
        comparison[folder,constraint_name[k]] = max(0,-min(x))/scaling[constraint_name[k]]

    # max MW
    # seperately
    x = [endpoint_constraints[i]['epc_mw'] for i in range(iters) if endpoint_constraints[i][constraint_name[k]] != 'error']
    # compute standard deviation
    std = np.std(x) 
    mu = np.mean(x)
    n = 100
    # remove outliers (not in interval +-n x std)
    x = [i for i in x if i >= mu-n*std and i <= mu+n*std]
    comparison[folder,'epc_mw_ub'] = max(0,max(x)-20)/scaling['epc_mw']
    
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
    comparison[folder,'max_scenario'] = max(tf,key=tf.get)
    comparison[folder,'avg_tf'] = avg_tf
    comparison[folder,'min_tf'] = min_tf
    comparison[folder,'min_scenario'] = min(tf,key=tf.get)
    
    comparison[folder,'max_regret'] = max([tf[i]-lb[i] for i in tf]) 
    comparison[folder,'avg_regret'] = avg_tf - comparison['baseline','avg_tf']
    comparison[folder,'min_regret'] = min([tf[i]-lb[i] for i in tf])
    
#scenarios[{key: value for key, value in comparison[folder,'scen'].items()}]
    comparison[folder,'t_ocp'] = [CPU_t[i][k,'ocp','cpu'] for i in CPU_t for k in range(1,24) if CPU_t[i] != 'error']
    comparison[folder,'t_mhe'] = [CPU_t[i][k,'mhe','cpu'] for i in CPU_t for k in range(1,24) if CPU_t[i] != 'error' ]
    try:
        comparison[folder,'t_cr'] = [CPU_t[i]['cr'] for i in CPU_t]
    except:
        pass
    
    


###############################################################################
############################# create plots ####################################
###############################################################################

# robustness level + performance
fig, ax1 = plt.subplots()
fig.set_size_inches(17.5/2.54, 10/2.54)
ind = np.arange(len(folders))
width = 1.0/8.0
bars = {}
xticks = [method[f] for f in folders]
#xspacing such that the label is centered
xspacing = [(ind[i] + ind[i+1])/2.0-1.5*width for i in range(len(ind)-1)]
xspacing.append((2*ind[-1]+1)/2.0-1.5*width)
i = 0.0

labels = {'T_max':r'$T^{max} \, [K]$','Tad_max':r'$T^{max}_{ad} \, [K]$','T_min':r'$T^{min} \, [K]$','epc_mw':r'$NAMW \, [\frac{g}{mol}]$', 'epc_PO_ptg':r'$unreac \, [10^{2} \cdot ppm]$','epc_unsat':r'$unsat \, [10^{-3} \cdot \frac{mol}{g}]$','max_tf':r'$t_f^{max}$','avg_tf':r'$t_f^{avg}$','min_tf':r'$t_f^{min}$'}
color = ['w','lightgrey','grey']
k = 0
for con in constraint_name:
    vals = []
    for folder in folders:
        vals.append(comparison[folder,con])
    vals = np.array(vals)#/max(abs(min(vals)),max(vals))*100
    bars[con] = ax1.bar(ind+width*i, vals,width,align='center',color = color[k], edgecolor='k', label=labels[con])
    i += 1
    k += 1

k = 0
for con in ['T_max','Tad_max','T_min']:
    vals = []
    for folder in folders:
        vals.append(comparison[folder,con])
    vals = np.array(vals)#/max(abs(min(vals)),max(vals))*100
    bars[con] = ax1.bar(ind+width*i, vals, width, label=labels[con],color = color[k], edgecolor='k', hatch="//")
    i += 1
    k += 1
ax1.set_xticks(xspacing)
ax1.set_xticklabels(xticks) #can hold text
ax1.set_ylabel('maximum constraint violation')
ax1.tick_params(axis='y',direction='in')
lgd = ax1.legend(loc ='lower center', ncol=3, fancybox=True, bbox_to_anchor=(0.5,-0.3))
ax2 = ax1.twinx()

color = ['k','k','k']
lstyle = ['-.','-','--']
k = 0
for t in ['max_tf','avg_tf','min_tf']:
    vals = []
    for folder in folders:
        vals.append(comparison[folder,t])
    ax2.plot(xspacing,vals,linestyle=lstyle[k],marker='o',color=color[k],label=labels[t])
    #ax2.plot([xspacing[0],xspacing[-1]],[comparison['baseline',t]]*2,linestyle='--', color='r')
    k += 1
ax2.set_ylim([0,750])
ax2.set_ylabel(r'$t_f$ [min]')
ax2.tick_params(axis='y',direction='in')
ax2.legend(bbox_to_anchor=(0.98,0.6))#loc='lower right')
fig.tight_layout()
plt.show()
fig.savefig('results/'+p1+'sp.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
fig.savefig('results/'+p1+'sp.svg',bbox_extra_artists=(lgd,), bbox_inches='tight')


# CPU times

fig,ax = plt.subplots()
# y-axis: % of instances solved in time t
# x-axis: % time t
style =   [('solid',               (0, ())),
           ('dotted',              (0, (1, 5))),
           ('dashed',              (0, (5, 5))),
           ('dashdotted',          (0, (3, 5, 1, 5))),
           ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
           ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),

             ('loosely dashed',      (0, (5, 10))),
             
             ('densely dashed',      (0, (5, 1))),
        
             ('loosely dashdotted',  (0, (3, 10, 1, 10))),
             
             ('densely dashdotted',  (0, (3, 1, 1, 1))),
        
             ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
             
             
             ('loosely dotted',      (0, (1, 10))),
             
             ('densely dotted',      (0, (1, 1))),]
k = 0
t_steps = np.linspace(0,1,2000)
for folder in folders: 
    aux = [100*sum(1.0 for entry in comparison[folder,'t_mhe'] if entry < t_steps[i])/len(comparison[folder,'t_mhe']) for i in range(len(t_steps))]
    ax.plot(t_steps,aux,linestyle=style[k][1],color='k',label=method[folder])
    k+=1
ax.legend()
ax.set_xlabel(r'$t_{CPU}$ [s]')
ax.set_ylabel('Percentage of instances solved [%]')
ax.tick_params(axis='both',direction='in')
fig.savefig('results/'+p1+'comp_times_mhe.pdf')
fig.savefig('results/'+p1+'comp_times_mhe.svg')

# NMPC
fig, ax = plt.subplots()
# y-axis: % of instances solved in time t
# x-axis: % time t
t_steps = np.linspace(0,20,2000)

k = 0
for folder in folders: 
    aux = [100*sum(1.0 for entry in comparison[folder,'t_ocp'] if entry < t_steps[i])/len(comparison[folder,'t_mhe']) for i in range(len(t_steps))]
    ax.plot(t_steps,aux,linestyle=style[k][1],color='k',label=method[folder])
    k += 1
ax.set_xlabel(r'$t_{CPU}$ [s]')
ax.set_ylabel('Percentage of instances solved [%]')
ax.tick_params(axis='both',direction='in')
ax.legend()
fig.savefig('results/'+p1+'comp_times_ocp.pdf')
fig.savefig('results/'+p1+'comp_times_ocp.svg')

###############################################################################
########################### baseline comparison ###############################
###############################################################################
print('min')
f = open('results/'+p1+'constraint_table.txt','w')
#f.write('min_regret' + '\n')
for folder in folders:
    print(folder, comparison[folder,'min_tf'] - lb[comparison[folder,'min_scenario']])
    print(folder, comparison[folder,'min_tf'] - comparison['baseline','min_tf'])
    print(folder, comparison[folder,'min_regret'])
    #f.write(method[folder] + ': ' + str(comparison[folder,'min_regret']) + '\n')
print('max')
#f.write('max_regret' + '\n')
for folder in folders:
    print(folder, comparison[folder,'max_tf'] - lb[comparison[folder,'max_scenario']])
    print(folder, comparison[folder,'max_tf'] - comparison['baseline','max_tf'])
    print(folder, comparison[folder,'max_regret'])
    #f.write(method[folder] + ': ' + str(comparison[folder,'max_regret']) + '\n')
    print(scenarios[comparison[folder,'max_scenario']])
print('avg')
#f.write('avg_regret' + '\n')
for folder in folders:
    print(folder, comparison[folder,'avg_tf'] - comparison['baseline','avg_tf'])
    #f.write(method[folder] + ': ' + str(comparison[folder,'avg_regret']) + '\n')
#f.write('\n' + '\n' + '\n')

for folder in folders:
    f.write(method[folder] + '\t & \t $\SI{' + str(comparison[folder,'max_regret']) + '}{}$ \t & \t $\SI{' \
            + str(comparison[folder,'avg_regret']) + '}{}$ \t & \t $\SI{' \
            + str(comparison[folder,'min_regret']) + '}{}$ \\\ \n')
f.write('\n' + '\n' + '\n')
###############################################################################
############################# create table ####################################
###############################################################################

for con in constraint_name+['epc_mw_ub']:
    f.write('#'*20+con+'#'*20 + '\n')
    for folder in folders:
        f.write(folder + ': ' + str(comparison[folder,con]) + '\n')
        
for con in ['T_max','Tad_max','T_min']:
    f.write('#'*20+con+'#'*20 + '\n')
    for folder in folders:   
        f.write(folder + ': ' + str(comparison[folder,con]) + '\n')

f.write('\n' + '\n' + '\n')

for folder in folders:
    #folder[folder.rfind('/')+1:]
    f.write(method[folder] + '\t & \t' + str(comparison[folder,'max_tf'])+ '\t & \t' + str(comparison[folder,'avg_tf']) + '\t & \t' + str(comparison[folder,'min_tf']) + '\\\ \n')

f.write('\n' + '\n' + '\n')

for folder in folders:
    #folder[folder.rfind('/')+1:]
    f.write(method[folder] + '\t & \t' + str(comparison[folder,'epc_PO_ptg']*scaling['epc_PO_ptg'])+ '\t & \t' + str(comparison[folder,'epc_unsat']*scaling['epc_unsat']) + '\t & \t' + str(comparison[folder,'epc_mw']*scaling['epc_mw']) + '\\\ \n')
f.close()
    
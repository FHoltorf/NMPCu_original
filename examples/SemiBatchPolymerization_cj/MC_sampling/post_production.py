#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:54:57 2018

@author: flemmingholtorf
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt # for 2d plots
from mpl_toolkits.mplot3d import Axes3D # for 3d histogram plots
import sys

#folders = ['online_estimation','multistage','backoff','standard']
p1 = '125grid/timeinvariant/standard/'
#p2 = 'finalfinal/timeinvariant/standard/'
folders = [p1+'nominal',p1+'SBBM',p1+'ms',p1+'SBSG',p1+'nominal_bo']#,p1+'SBSG1stage']
directory = 'results/125grid/' # save overall plots here
method = {p1+'nominal':'NMPC',
          p1+'nominal_bo':'NMPC-BO',
          p1+'SBBM':'NMPC-SBBM',
          p1+'ms':'msNMPC',
          p1+'SBSG':'msNMPC-SBSG'}
          #p1+'SBSG1stage':'ms-SBSG2'}
#method = {'cj/MHE/ideal/tiv-few_meas/parnoise':'iNMPC-pn','cj/MHE/ideal/tiv-few_meas/parest-adapt':'iNMPC-adapt','cj/MHE/ideal/tiv-few_meas/parest-noadapt':'iNMPC','cj/MHE/multistage/tiv-few_meas/parest-noadapt':'msNMPC','cj/MHE/multistage/tiv-few_meas/stadapt':'msNMPC ST','cj/MHE/SBBM/tiv-few_meas/parest-adapt':'rNMPC-adapt'}
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
    
    f = open(path + 'scenarios.pckl','rb')
    scenarios = pickle.load(f)
    f.close()
    
    f = open(path + 'uncertainty_realization.pckl','rb')
    uncertainty_realization = pickle.load(f)
    f.close()
    
    constraint_name = []
    iters = len(tf)
    for constraint in endpoint_constraints[0]:
        if constraint == 'feasible':
            continue
        constraint_name.append(constraint)
    
    if folder == 'standard':
        for i in range(iters):
            try:
                endpoint_constraints[i]['epc_unsat'] = 0.032 - endpoint_constraints[i]['epc_unsat']
            except TypeError:
                continue
    else:
        for i in range(iters):
            try:
                endpoint_constraints[i]['epc_unsat'] = 0.032 - endpoint_constraints[i]['epc_unsat']
            except TypeError:
                continue
        
    for i in range(iters):
        try:
            endpoint_constraints[i]['epc_PO_ptg'] = 120.0 - endpoint_constraints[i]['epc_PO_ptg'] 
        except TypeError:
            continue
        
    for i in range(iters):
        try:
            endpoint_constraints[i]['epc_mw'] = 949.5 + endpoint_constraints[i]['epc_mw'] 
        except TypeError:
            continue
       
    #unit = {'epc_PO_ptg' : ' [PPM]', 'epc_unsat' : ' [mol/g PO]', 'epc_mw' : ' [g/mol]'}
    xlabel = {'epc_PO_ptg' : 'Unreacted monomer [PPM]', 'epc_unsat' : r'Unsaturated by-product $[\frac{mol}{g_{PO}}]$', 'epc_mw' : r'NAMW [$\frac{g}{mol}$]'}
    #axes = {'epc_PO_ptg' : [-80.0,120.0,0.0,75.0], 'epc_unsat' : [-30.0,70.0,0.0,35.0], 'epc_mw' : [-0.5,3.0,0.0,35.0],'tf':[320.0,500.0,0.0,35.0]}
    #axes = {'epc_PO_ptg' : [0.0,240.0], 'epc_unsat' : [0.029,0.0345], 'epc_mw' : [948.8,952.5],'tf':[320.0,500.0]}
    axes = {'epc_PO_ptg' : [0.0,450], 'epc_unsat' : [0.025,0.041], 'epc_mw' : [947.0,970.0],'tf':[240.0,720.0]}
    set_points = {'epc_PO_ptg' : 120, 'epc_unsat' : 0.032, 'epc_mw' : 949.5} 
    feasible_region = {'epc_PO_ptg' : 'l', 'epc_unsat' : 'l', 'epc_mw' : 'r'}
    # enpoint constraints 
    for k in range(3):
        color[k]
        x = [endpoint_constraints[i][constraint_name[k]] for i in range(iters) if endpoint_constraints[i][constraint_name[k]] != 'error']
        print('wc error ', constraint_name[k], max(x))
        # compute standard deviation
        std = np.std(x) 
        mu = np.mean(x)
        n = 100
        # remove outliers (not in interval +-n x std)
        x = [i for i in x if i >= mu-n*std and i <= mu+n*std]
        comparison[folder,constraint_name[k]] = x
        plt.figure(k)
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(x, 'auto', normed=None, facecolor=color[k], edgecolor='black', alpha=1.0)
        length = max(n)
        ax.plot([set_points[constraint_name[k]],set_points[constraint_name[k]]],[0.0,1.1*length],color='red',linestyle='dashed',linewidth=2)
        ax.set_xlim(axes[constraint_name[k]])
        # add label for feasible/infeasible regions
        if feasible_region[constraint_name[k]] == 'l':
            ax.text((set_points[constraint_name[k]]+axes[constraint_name[k]][0])/2,1.05*length,'feasible', fontweight='bold', horizontalalignment='center', verticalalignment='center')
            ax.text((axes[constraint_name[k]][1]+set_points[constraint_name[k]])/2,1.05*length,'infeasible', fontweight='bold', horizontalalignment='center', verticalalignment='center')
        else:
            ax.text((set_points[constraint_name[k]]+axes[constraint_name[k]][0])/2,1.05*length,'infeasible', fontweight='bold', horizontalalignment='center', verticalalignment='center')
            ax.text((axes[constraint_name[k]][1]+set_points[constraint_name[k]])/2,1.05*length,'feasible', fontweight='bold', horizontalalignment='center', verticalalignment='center')
        plt.xlabel(xlabel[constraint_name[k]])
        plt.ylabel('Frequency [-]')
        fig.savefig(path + constraint_name[k] +'.pdf')
#    
    
    # compute final time histogram
    plt.figure(4)
    x = [tf[i] for i in range(iters) if endpoint_constraints[i]['feasible'] != 'crashed']
    comparison[folder,'tf'] = x
    fig, ax = plt.subplots()
    #plt.hist(x,int(np.ceil(iters**0.5)), normed=None, facecolor='purple', edgecolor='black', alpha=1.0) 
    ax.hist(x,'auto', normed=None, facecolor='purple', edgecolor='black', alpha=1.0) 
    ax.set_xlim(axes['tf'])
    plt.xlabel('Final batch time [min]')
    plt.ylabel('Frequency [-]')
    fig.savefig(path + 'tf.pdf')
    
    # compute average tf
    tf_bar = sum(tf[i] for i in range(iters) if endpoint_constraints[i]['feasible'] != 'crashed')/sum(1 for i in range(iters) if endpoint_constraints[i]['feasible'] != 'crashed')
    endpoint_constraints['tf_avg'] = tf_bar
    
    
    # path constraints
    #heat_removal = {}
    T={}
    t = {}
    Tad = {}
    for i in path_constraints: # loop over all runs
        if path_constraints[i] =='error':
            continue

        t[i] = []
        Tad[i] = []
        T[i] = []
        #heat_removal[i] = []
        for fe in range(1,25):
            for cp in range(1,4):        
                #heat_removal[i].append(path_constraints[i]['heat_removal',(fe,(cp,))]*92048.0/60.0)
                T[i].append(path_constraints[i]['T',(fe,(cp,))]*100.0)
                Tad[i].append(path_constraints[i]['Tad',(fe,(cp,))]*100.0)
                if fe > 1:
                    t[i].append(t[i][-cp]+path_constraints[i]['tf',(fe,cp)])
                else:
                    t[i].append(path_constraints[i]['tf',(fe,cp)])
        
    max_tf = max([tf[i] for i in tf if endpoint_constraints[i]['feasible'] != 'crashed'])   
    min_tf = min([tf[i] for i in tf if endpoint_constraints[i]['feasible'] != 'crashed'])
    plt.figure(5)
    fig, ax = plt.subplots()
    for i in Tad:
        ax.plot(t[i],Tad[i], color='grey')
    ax.plot([0,max_tf],[4.4315e2,4.4315e2], color='red', linestyle='dashed')
    ax.tick_params(axis='both',direction='in')
    plt.xlabel(r'$t$ [min]')
    plt.ylabel(r'$T_{ad}$ [K]')
    fig.savefig(path+'Tad.pdf')
    
    # plot the enclosing hull
    # much much more difficult than i thought!
    plt.figure(6)
    fig, ax = plt.subplots()
    for i in Tad:
        #ax.plot(t[i],heat_removal[i], color='grey')
        ax.plot(t[i],T[i], color='grey')
    #ax.plot([0,max_tf],[2200,2200], color='red', linestyle='dashed') #1.43403,1.43403
    ax.plot([0,max_tf],[423.15,423.15], color='red', linestyle='dashed') #1.43403,1.43403
    ax.plot([0,max_tf],[373.15,373.15], color='red', linestyle='dashed') #1.43403,1.43403
    ax.tick_params(axis='both',direction='in')
    plt.xlabel(r'$t$ [min]')
    plt.ylabel(r'$T$ [K]')
    fig.savefig(path+'T.pdf')
    #plt.ylabel('Q [kW]')
    #fig.savefig(path+'heat_removal.pdf')

    fes = 0
    infes = 0
    
    for i in range(iters):
        # problem is feasible
        if endpoint_constraints[i]['feasible'] == True:
            fes += 1
        elif endpoint_constraints[i]['feasible'] == False:
            infes += 1
    sizes = [fes, infes, iters-fes-infes]
    
    plt.figure(7)
    fig, ax = plt.subplots()
    explode = (0.0, 0.1, 0.0) 
    wedges= ax.pie(sizes,explode,labels=['feasible','infeasible','crashed'], autopct='%1.1f%%',shadow=True)
    for w in wedges[0]:
        w.set_edgecolor('black')
    plt.axis('equal')
    fig.savefig(path + 'feas.pdf')
    
    # average economic performance:
    print('best performance', min_tf, '[min]')
    print('worst_performance', max_tf, '[min]')
    print('avg_performance',sum(tf[i] for i in tf if endpoint_constraints[i]['feasible'] != 'crashed')/sum(1 for i in tf if endpoint_constraints[i]['feasible'] != 'crashed'),'[min]')
    
    # computational performance
    comparison[folder,'t_ocp'] = [CPU_t[i][k,'ocp','cpu'] for i in CPU_t for k in range(1,24) if CPU_t[i] != 'error']
    comparison[folder,'t_mhe'] = [CPU_t[i][k,'mhe','cpu'] for i in CPU_t for k in range(1,24) if CPU_t[i] != 'error' ]
    try:
        comparison[folder,'t_cr'] = [CPU_t[i]['cr'] for i in CPU_t]
    except:
        pass
    
# plot the comparison distribution plots
# colors
spacing = list(np.linspace(10*(len(folders)-1),0,len(folders)))
colors = ['r', 'g', 'b', 'y','c','m','k']
yticks = [method[f] for f in folders]
for k in range(3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis._axinfo['label']['space_factor'] = 2.0
    ax.yaxis._axinfo['label']['space_factor'] = 2.0
    ax.zaxis._axinfo['label']['space_factor'] = 2.0
    #nbins = 50
    i = 0
    for c, z in zip(colors,spacing):
        f = folders[i]
        i += 1
        #ys = 100*(np.array(comparison[f,constraint_name[k]])-np.array(set_points[constraint_name[k]]))/np.array(set_points[constraint_name[k]])
        ys = comparison[f,constraint_name[k]]
        hist, bins = np.histogram(ys,'auto')#, bins=nbins)
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=z, zdir='y', width = 0.8*(bins[2]-bins[1]), color=c, ec='black', alpha=0.8)#
    #plot setpoint
    ax.plot([set_points[constraint_name[k]],set_points[constraint_name[k]]],[0,0],[min(spacing),max(spacing)],zdir='y',c='r',linestyle='dashed')
    #ax.set_xlim(axes[constraint_name[k]])
    ax.set_xlabel(xlabel[constraint_name[k]],labelpad=15.0)
    ax.set_yticks(spacing)
    ax.set_yticklabels(yticks) #can hold text
    ax.set_ylabel('')
    ax.set_zlabel('Frequency [-]')
    fig.autofmt_xdate()
    ax.grid(False)
    fig.savefig(directory+constraint_name[k]+'cp.pdf')
    
    
# tf

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
i = 0
for c, z in zip(colors,spacing):
    f = folders[i]
    i += 1
    #ys = 100*(np.array(comparison[f,constraint_name[k]])-np.array(set_points[constraint_name[k]]))/np.array(set_points[constraint_name[k]])
    ys = comparison[f,'tf']
    hist, bins = np.histogram(ys,'auto')#, bins=nbins)
    xs = (bins[:-1] + bins[1:])/2
    ax.bar(xs, hist, zs=z, zdir='y', width = 0.8*(bins[2]-bins[1]), color=c, ec='black', alpha=0.8)#
ax.set_xlabel('tf [min]',labelpad=15.0)
ax.set_yticks(spacing)
ax.set_yticklabels(yticks) #can hold text
ax.set_ylabel('')
ax.set_zlabel('Frequency [-]')
fig.autofmt_xdate()
ax.grid(False)
fig.savefig(directory+'tf_cp.pdf')

"""
# CPU times
# MHE
fig = plt.figure()
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
t_steps = np.linspace(0,4,50)
for folder in folders: 
    aux = [100*sum(1.0 for entry in comparison[folder,'t_mhe'] if entry < t_steps[i])/len(comparison[folder,'t_mhe']) for i in range(len(t_steps))]
    plt.plot(t_steps,aux,linestyle=style[k][1],color='k',label=method[folder])
    k+=1
plt.legend()
plt.xlabel('Execution time (real time) [s]')
plt.ylabel('Percentage of instances solved [%]')
fig.savefig(path+'comp_times_mhe.pdf')

# CPU times
# NMPC
fig = plt.figure()
# y-axis: % of instances solved in time t
# x-axis: % time t
t_steps = np.linspace(0,60,100)

k = 0
for folder in folders: 
    aux = [100*sum(1.0 for entry in comparison[folder,'t_ocp'] if entry < t_steps[i])/len(comparison[folder,'t_mhe']) for i in range(len(t_steps))]
    plt.plot(t_steps,aux,linestyle=style[k][1],color='k',label=method[folder])
    k += 1
plt.xlabel('Execution time (real time) [s]')
plt.ylabel('Percentage of instances solved [%]')
plt.legend()
fig.savefig(path+'comp_times_ocp.pdf')
#NMPC
"""


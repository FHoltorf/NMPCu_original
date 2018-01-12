#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:54:57 2018

@author: flemmingholtorf
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

#folders = ['online_estimation','multistage','backoff','standard']
folders = ['backoff']
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
    axes = {'epc_PO_ptg' : [0.0,450], 'epc_unsat' : [0.025,0.041], 'epc_mw' : [947.0,962.0],'tf':[320.0,500.0]}
    set_points = {'epc_PO_ptg' : 120, 'epc_unsat' : 0.032, 'epc_mw' : 949.5} 
    feasible_region = {'epc_PO_ptg' : 'l', 'epc_unsat' : 'l', 'epc_mw' : 'r'}
    # enpoint constraints 
    for k in range(3):
        color[k]
        x = [endpoint_constraints[i][constraint_name[k]] for i in range(iters) if endpoint_constraints[i][constraint_name[k]] != 'error']
        # compute standard deviation
        std = np.std(x) 
        mu = np.mean(x)
        # remove outliers (not in interval +-3 x std)
        x = [i for i in x if i >= mu-3*std and i <= mu+3*std]
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
    
    
    # compute final time histogram
    plt.figure(4)
    x = [tf[i] for i in range(iters) if endpoint_constraints[i]['feasible'] != 'crashed']
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
    heat_removal = {}
    t = {}
    Tad = {}
    for i in path_constraints: # loop over all runs
        if path_constraints[i] =='error':
            continue
        heat_removal[i] = []
        t[i] = []
        Tad[i] = []
        for fe in range(1,25):
            for cp in range(1,4):        
                heat_removal[i].append(path_constraints[i]['heat_removal',(fe,(cp,))]*92048.0/60.0)
                Tad[i].append(path_constraints[i]['Tad',(fe,(cp,))]*100)
                if fe > 1:
                    t[i].append(t[i][-cp]+path_constraints[i]['tf',(fe,cp)])
                else:
                    t[i].append(path_constraints[i]['tf',(fe,cp)])
        
        
    max_tf = max([tf[i] for i in tf if endpoint_constraints[i]['feasible'] != 'crashed'])    
    plt.figure(5)
    fig, ax = plt.subplots()
    for i in Tad:
        ax.plot(t[i],Tad[i], color='grey')
    ax.plot([0,max_tf],[4.6315e2,4.6315e2], color='red', linestyle='dashed')
    plt.xlabel('t [min]')
    plt.ylabel(r'$T_{ad}$ [K]')
    fig.savefig(path+'Tad.pdf')
    
    # plot the enclosing hull
    # much much more difficult than i thought!
    plt.figure(6)
    fig, ax = plt.subplots()
    for i in Tad:
        ax.plot(t[i],heat_removal[i], color='grey')
    ax.plot([0,max_tf],[2200,2200], color='red', linestyle='dashed') #1.43403,1.43403
    plt.xlabel('t [min]')
    plt.ylabel('Q [kW]')
    fig.savefig(path+'heat_removal.pdf')
    
    
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
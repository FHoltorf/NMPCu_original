#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:54:57 2018

@author: flemmingholtorf
"""
from mpl_toolkits.mplot3d import Axes3D # for 3d histogram plots
import numpy as np
import matplotlib.pyplot as plt # for 2d plots
import sys, pickle

path = 'results/125grid/standard/'
folders = ['nominal','ms','SBSG','SBBM']#,path+'SBBM',path+'ms',path+'SBSG']
method = {'nominal':'NMPC',
          'nominal_bo':'NMPC-BO',
          'SBBM':'NMPC-SBBM',
          'ms':'msNMPC',
          'SBSG':'msNMPC-SBSG'}
comparison = {}
comparison_wc = {}
scaling = {'epc_mw':1e0, 'epc_PO_ptg':1e3,'epc_unsat':1e-3}
setpoint = {'epc_mw':949.5, 'epc_PO_ptg':120.0,'epc_unsat':0.032}
for folder in folders: 
    print(folder)
    directory = path + folder+'/' 
    color = ['green','yellow','blue']
    
    raw_data = pickle.load(open(directory+'results.pckl','rb'))    
    endpoint_constraints = raw_data['epc']
    path_constraints = raw_data['pc']
    tf = raw_data['tf']
    CPU_t = raw_data['t_CPU']
    scenarios = raw_data['scenarios']
        
    constraint_name = []
    iters = len(tf)
    for constraint in endpoint_constraints[0]:
        constraint_name.append(constraint)
    
    for i in range(iters):
        try:
            endpoint_constraints[i]['epc_unsat'] = setpoint['epc_unsat'] - endpoint_constraints[i]['epc_unsat']
        except TypeError:
            continue
            
    for i in range(iters):
        try:
            endpoint_constraints[i]['epc_PO_ptg'] = setpoint['epc_PO_ptg'] - endpoint_constraints[i]['epc_PO_ptg'] 
        except TypeError:
            continue
        
    for i in range(iters):
        try:
            endpoint_constraints[i]['epc_mw'] = setpoint['epc_mw'] + endpoint_constraints[i]['epc_mw'] 
        except TypeError:
            continue
       
    xlabel = {'epc_PO_ptg' : 'Unreacted monomer [PPM]', 'epc_unsat' : r'Unsaturated by-product $[\frac{mol}{g_{PO}}]$', 'epc_mw' : r'NAMW [$\frac{g}{mol}$]'}
    #axes = {'epc_PO_ptg' : [-80.0,120.0,0.0,75.0], 'epc_unsat' : [-30.0,70.0,0.0,35.0], 'epc_mw' : [-0.5,3.0,0.0,35.0],'tf':[320.0,500.0,0.0,35.0]}
    #axes = {'epc_PO_ptg' : [0.0,240.0], 'epc_unsat' : [0.029,0.0345], 'epc_mw' : [948.8,952.5],'tf':[320.0,500.0]}
    axes = {'epc_PO_ptg' : [0.0,450], 'epc_unsat' : [0.025,0.041], 'epc_mw' : [947.0,970.0],'tf':[240.0,720.0]}
    set_points = {'epc_PO_ptg' : 120, 'epc_unsat' : 0.032, 'epc_mw' : 949.5} 
    feasible_region = {'epc_PO_ptg' : 'l', 'epc_unsat' : 'l', 'epc_mw' : 'r'}
    
    # enpoint constraints 
    for k in range(3):
        color[k]
        x = [endpoint_constraints[i][constraint_name[k]] for i in range(iters) if endpoint_constraints[i] != 'crashed']
        print('wc error ', constraint_name[k], max(x))
        # compute standard deviation
        std = np.std(x) 
        mu = np.mean(x)
        n = 1e10        # remove outliers (not in interval +-n x std)
        x = [i for i in x if i >= mu-n*std and i <= mu+n*std]
        comparison[folder,constraint_name[k]] = x
        comparison_wc[folder,constraint_name[k]] = max(0,max(x) - setpoint[constraint_name[k]])/scaling[constraint_name[k]] if constraint_name[k] != 'epc_mw' else max(0,max(x) - setpoint[constraint_name[k]] - 20)/scaling[constraint_name[k]] 
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
        fig.savefig(directory + constraint_name[k] +'.pdf')
    
    # min MW
    x = [endpoint_constraints[i]['epc_mw'] for i in range(iters) if endpoint_constraints[i] != 'crashed']
    # compute standard deviation
    std = np.std(x) 
    mu = np.mean(x)
    n = 100
    # remove outliers (not in interval +-n x std)
    x = [i for i in x if i >= mu-n*std and i <= mu+n*std]
    comparison_wc[folder,'epc_mw'] = max(comparison_wc[folder,'epc_mw'], (setpoint['epc_mw'] - min(x))/scaling['epc_mw'])
    
    
    
    # compute final time histogram
    plt.figure(4)
    x = [tf[i] for i in range(iters) if endpoint_constraints[i] != 'crashed']
    comparison[folder,'tf'] = x
    fig, ax = plt.subplots()
    #plt.hist(x,int(np.ceil(iters**0.5)), normed=None, facecolor='purple', edgecolor='black', alpha=1.0) 
    ax.hist(x,'auto', normed=None, facecolor='purple', edgecolor='black', alpha=1.0) 
    ax.set_xlim(axes['tf'])
    plt.xlabel('Final batch time [min]')
    plt.ylabel('Frequency [-]')
    fig.savefig(directory + 'tf.pdf')
    
    
    # path constraints
    T={}
    t = {}
    Tad = {}
    Tad_max = -1e8
    T_max = -1e8
    T_min = 1e8

    for i in path_constraints: # loop over all runs
        if path_constraints[i] == 'crashed':
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
                if T_max < path_constraints[i]['T',(fe,(cp,))]*100.0:
                    T_max = path_constraints[i]['T',(fe,(cp,))]*100.0
                if Tad_max < path_constraints[i]['Tad',(fe,(cp,))]*100.0:
                    Tad_max = path_constraints[i]['Tad',(fe,(cp,))]*100.0
                if T_min > path_constraints[i]['T',(fe,(cp,))]*100.0:
                    T_min = path_constraints[i]['T',(fe,(cp,))]*100.0
        
    max_tf = max([tf[i] for i in tf if tf[i] != 'crashed'])   
    avg_tf = sum(tf[i] for i in tf if tf[i] != 'crashed')/sum(1 for i in tf if tf[i] != 'crashed')
    min_tf = min([tf[i] for i in tf if tf[i] != 'crashed'])
    
    plt.figure(5)
    fig, ax = plt.subplots()
    for i in Tad:
        ax.plot(t[i],Tad[i], color='grey')
    ax.plot([0,max_tf],[4.4315e2,4.4315e2], color='red', linestyle='dashed')
    ax.tick_params(axis='both',direction='in')
    plt.xlabel(r'$t$ [min]')
    plt.ylabel(r'$T_{ad}$ [K]')
    fig.savefig(directory +'Tad.pdf')
    
    plt.figure(6)
    fig, ax = plt.subplots()
    for i in Tad:
        ax.plot(t[i],T[i], color='grey')
    ax.plot([0,max_tf],[423.15,423.15], color='red', linestyle='dashed') #1.43403,1.43403
    ax.plot([0,max_tf],[373.15,373.15], color='red', linestyle='dashed') #1.43403,1.43403
    ax.tick_params(axis='both',direction='in')
    plt.xlabel(r'$t$ [min]')
    plt.ylabel(r'$T$ [K]')
    fig.savefig(directory +'T.pdf')

    """
    fes = 0
    infes = 0
    for i in endpoint_constraints:
        endpoint_constraints[i]
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
    fig.savefig(directory + 'feas.pdf')
    """    
    # average economic performance:
    print('best performance', min_tf, '[min]')
    print('worst_performance', max_tf, '[min]')
    print('avg_performance',sum(tf[i] for i in tf if endpoint_constraints[i] != 'crashed')/sum(1 for i in tf if endpoint_constraints[i] != 'crashed'),'[min]')
    
    # computational performance
    comparison_wc[folder,'T_max'] = max(0,T_max - 423.15)
    comparison_wc[folder,'Tad_max'] = max(0,Tad_max - 443.15)
    comparison_wc[folder,'T_min'] = max(0,-T_min + 373.15)
    comparison_wc[folder,'max_tf'] = max_tf
    comparison_wc[folder,'max_scenario'] = max(tf,key=tf.get)
    comparison_wc[folder,'avg_tf'] = avg_tf
    comparison_wc[folder,'min_tf'] = min_tf
    comparison_wc[folder,'min_scenario'] = min(tf,key=tf.get)
    try:
        comparison_wc[folder,'max_regret'] = max([tf[i]-lb[i] for i in tf]) 
        comparison_wc[folder,'avg_regret'] = avg_tf - comparison['baseline','avg_tf']
        comparison_wc[folder,'min_regret'] = min([tf[i]-lb[i] for i in tf])
    except:
        pass

    comparison_wc[folder,'lsmhe'] = [sum(CPU_t[i][k][1][l].ru_utime - CPU_t[i][k][0][l].ru_utime for l in [1]) for i in CPU_t for k in CPU_t[i] if k[0] == 'lsmhe']
    comparison_wc[folder,'olnmpc'] = [sum(CPU_t[i][k][1][l].ru_utime - CPU_t[i][k][0][l].ru_utime for l in [1]) for i in CPU_t for k in CPU_t[i] if k[0] == 'olnmpc']
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
        ys = comparison[f,constraint_name[k]]
        hist, bins = np.histogram(ys,'auto')#, bins=nbins)
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=z, zdir='y', width = 0.8*(bins[2]-bins[1]), color=c, ec='black', alpha=0.8)#
    #plot setpoint
    ax.plot([set_points[constraint_name[k]],set_points[constraint_name[k]]],[0,0],[min(spacing),max(spacing)],zdir='y',c='r',linestyle='dashed')
    ax.set_xlabel(xlabel[constraint_name[k]],labelpad=15.0)
    ax.set_yticks(spacing)
    ax.set_yticklabels(yticks) #can hold text
    ax.set_ylabel('')
    ax.set_zlabel('Frequency [-]')
    fig.autofmt_xdate()
    ax.grid(False)
    fig.savefig(path+constraint_name[k]+'cp.pdf')
        
# tf
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
i = 0
for c, z in zip(colors,spacing):
    f = folders[i]
    i += 1
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
fig.savefig(path+'tf_cp.pdf')


""" summary visualization """
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

labels = {'T_max':r'$T^{max} \, [K]$','Tad_max':r'$T^{max}_{ad} \, [K]$','T_min':r'$T^{min} \, [K]$','epc_mw':r'$NAMW \, [\frac{g}{mol}]$', 'epc_PO_ptg':r'$unreac \, [10^{2} \cdot PPM]$','epc_unsat':r'$unsat \, [10^{-3} \cdot \frac{mol}{g_{PO}}]$','max_tf':r'$t_f^{max}$','avg_tf':r'$t_f^{avg}$','min_tf':r'$t_f^{min}$'}
color = ['w','lightgrey','grey']
k = 0
for con in constraint_name:
    vals = []
    for folder in folders:
        vals.append(comparison_wc[folder,con])
    vals = np.array(vals)#/max(abs(min(vals)),max(vals))*100
    bars[con] = ax1.bar(ind+width*i, vals,width,align='center',color = color[k], edgecolor='k', label=labels[con])
    i += 1
    k += 1

k = 0
for con in ['T_max','Tad_max','T_min']:
    vals = []
    for folder in folders:
        vals.append(comparison_wc[folder,con])
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
        vals.append(comparison_wc[folder,t])
    ax2.plot(xspacing,vals,linestyle=lstyle[k],marker='o',color=color[k],label=labels[t])
    #ax2.plot([xspacing[0],xspacing[-1]],[comparison['baseline',t]]*2,linestyle='--', color='r')
    k += 1
ax2.set_ylim([0,750])
ax2.set_ylabel(r'$t_f$ [min]')
ax2.tick_params(axis='y',direction='in')
ax2.legend(bbox_to_anchor=(0.98,0.6))#loc='lower right')
fig.tight_layout()
plt.show()
fig.savefig(path+'sp.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')


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
    aux = [100*sum(1.0 for entry in comparison_wc[folder,'lsmhe'] if entry < t_steps[i])/len(comparison_wc[folder,'lsmhe']) for i in range(len(t_steps))]
    ax.plot(t_steps,aux,linestyle=style[k][1],color='k',label=method[folder])
    k+=1
ax.legend()
ax.set_xlabel(r'$t_{CPU}$ [s]')
ax.set_ylabel('Percentage of instances solved [%]')
ax.tick_params(axis='both',direction='in')
fig.savefig(path+'comp_times_mhe.pdf')

# NMPC
fig, ax = plt.subplots()
# y-axis: % of instances solved in time t
# x-axis: % time t
t_steps = np.linspace(0,20,2000)

k = 0
for folder in folders: 
    aux = [100*sum(1.0 for entry in comparison_wc[folder,'olnmpc'] if entry < t_steps[i])/len(comparison_wc[folder,'olnmpc']) for i in range(len(t_steps))]
    ax.plot(t_steps,aux,linestyle=style[k][1],color='k',label=method[folder])
    k += 1
ax.set_xlabel(r'$t_{CPU}$ [s]')
ax.set_ylabel('Percentage of instances solved [%]')
ax.tick_params(axis='both',direction='in')
ax.legend()
fig.savefig(path+'comp_times_ocp.pdf')

###############################################################################
########################### baseline comparison ###############################
###############################################################################
"""
print('min')
f = open('results/'+path+'constraint_table.txt','w')
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
"""
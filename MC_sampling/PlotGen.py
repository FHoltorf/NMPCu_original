import numpy as np
from math import *
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Plotting Monte Carlo Simulation Results
sample_size = 50
unit = {'epc_PO_ptg' : r' [PPM]', 'epc_unsat' : r' $\left[\frac{mmol}{g PO}\right]$', 'epc_mw' : r' $\left[\frac{g}{mol}\right]$'}
epc = ['epc_PO_ptg','epc_unsat','epc_mw']
color = ['green','red','blue','purple']

tf = {}
res = {}

f = open('results/perturbed initialconditions/classic/Dec12/sampling_results.pckl', 'rb')
res['classic'] = pickle.load(f)
f.close()

f = open('results/perturbed initialconditions/open loop/sampling_results.pckl', 'rb')
res['open_loop'] = pickle.load(f)
f.close()

f = open('results/perturbed initialconditions/classic/Dec12/final_times.pckl', 'rb')
tf['classic'] = pickle.load(f)
f.close()

#f = open('results/Multistage/sampling_results.pckl', 'rb')
#res['multistage']= pickle.load(f)
#f.close()


# distributional plots
l = 0
for c in epc:
    plt.figure(l)
    k = 0
    for key in res:
        x = [res[key][i][c] for i in range(sample_size) if res[key][i][c] != 'error' and res[key][i][c] > -1000]
        if k == 0:
            n, bins, patches = plt.hist(x,int(ceil(sample_size**0.5)), normed=True, facecolor=color[k], edgecolor='black', alpha=0.5, label = key) 
        else:
            n, bins, patches = plt.hist(x,int(ceil(sample_size**0.5)), normed=True, facecolor=color[k], edgecolor='black', alpha=0.5, label = key)
        plt.xlabel(c  + unit[c])
        plt.ylabel('relative frequency [-]')
        plt.legend()
        k += 1
    plt.figure(l).savefig(c +'.pdf')
    l += 1    

# individual 3D scatter plots
k = 0
for key in res:
    fig = plt.figure(l)
    ax = fig.add_subplot(111, projection='3d')
    xsinfeas = {}
    xsfeas = {}
    for c in epc:
        xsinfeas[c] = [res[key][i][c] for i in range(sample_size) if res[key][i][c] != 'error' and not(res[key][i]['feasible']) and res[key][i]['epc_PO_ptg'] > -1000]
        xsfeas[c] = [res[key][i][c] for i in range(sample_size) if res[key][i][c] != 'error' and res[key][i]['feasible'] and res[key][i]['epc_PO_ptg'] > -1000]
    ax.scatter(xsinfeas[epc[0]],xsinfeas[epc[1]],xsinfeas[epc[2]], c=color[k], marker='o', depthshade=True, edgecolor = 'black')
    ax.scatter(xsfeas[epc[0]],xsfeas[epc[1]],xsfeas[epc[2]], c=color[k], marker='P', depthshade=True, edgecolor = 'black', label=key)
    ax.set_xlabel(epc[0])
    ax.set_ylabel(epc[1])
    ax.set_zlabel(epc[2])
    ax.legend()
    ax.view_init(30, 45)
    fig.savefig(key +'_scatter.pdf')
    l += 1  
    k += 1
    
# single 3D scatter plot for comparison
k = 0
fig = plt.figure(l)
ax = fig.add_subplot(111, projection='3d')
for key in res:
    xsinfeas = {}
    xsfeas = {}
    for c in epc:
        xsinfeas[c] = [res[key][i][c] for i in range(sample_size) if res[key][i][c] != 'error' and not(res[key][i]['feasible']) and res[key][i]['epc_PO_ptg'] > -1000]
        xsfeas[c] = [res[key][i][c] for i in range(sample_size) if res[key][i][c] != 'error' and res[key][i]['feasible'] and res[key][i]['epc_PO_ptg'] > -1000]
    ax.scatter(xsinfeas[epc[0]],xsinfeas[epc[1]],xsinfeas[epc[2]], c=color[k], marker='o', depthshade=True, edgecolor = 'black')
    ax.scatter(xsfeas[epc[0]],xsfeas[epc[1]],xsfeas[epc[2]], c=color[k], marker='P', depthshade=True, edgecolor = 'black', label=key)
    # plot plains indicating what is feasible
    k += 1
ax.legend()
ax.set_xlabel(epc[0])
ax.set_ylabel(epc[1])
ax.set_zlabel(epc[2])
ax.view_init(30, 45)
fig.savefig('combined_scatter.pdf')   
l += 1

k = 0
for key in res:
    plt.figure(l-1-2+k).axes[0].set_xlim(ax.get_xlim())
    plt.figure(l-1-2+k).axes[0].set_ylim(ax.get_ylim())
    plt.figure(l-1-2+k).axes[0].set_zlim(ax.get_zlim())
    plt.figure(l-1-2+k).savefig(key +'_scatter.pdf')
    k+=1

# final time distributation
plt.figure(l)
k = 0
for key in tf:    
    x = [tf[key][i] for i in range(sample_size) if tf[key][i] != 'error']
    plt.hist(x,int(ceil(sample_size**0.5)), normed=True, facecolor=color[k], edgecolor='black', alpha=0.5, label = key)
    k += 1
plt.xlabel('tf [min]')
plt.ylabel('relative frequency [-]')
plt.legend()
plt.figure(l).savefig('tf_distribution.pdf')
l += 1

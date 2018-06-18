#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 07:56:11 2018

@author: flemmingholtorf
"""
from __future__ import print_function
from main.mods.SemiBatchPolymerization.mod_class_roc import SemiBatchPolymerization
from main.dync.MHEGen import MHEGen
from main.examples.SemiBatchPolymerization.noise_characteristics import * 
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# don't write messy bytecode files
# might want to change if ran multiple times for performance increase
# sys.dont_write_bytecode = True 

# discretization parameters:
nfe, ncp = 24, 3 # Radau nodes assumed in code
# state variables:
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)], "T":[()], "T_cw":[()]}
# corrupted state variables:
x_noisy = []
# output variables:
y_vars = {"PO":[()], "Y":[()], "MY":[()], 'T':[()], 'T_cw':[()]}
# controls:
u = ["u1", "u2"]
u_bounds = {"u1": (0.0,0.4), "u2": (0.0, 3.0)} 
# uncertain parameters:
p_noisy = {"A":[('p',),('i',)],'kA':[()]}
# noisy initial conditions:
noisy_ics = {'PO_ic':[()],'T_ic':[()],'MY_ic':[()],'MX_ic':[(1,)]}
# initial uncertainty set description (hyperrectangular):
alpha = {('A', ('i',)):0.2,('A', ('p',)):0.2,('kA',()):0.2}#,
            #('PO_ic',()):0.02,('T_ic',()):0.005,
            #('MY_ic',()):0.01,('MX_ic',(1,)):0.002}
# time horizon bounds:
tf_bounds = [10.0*24/nfe, 30.0*24/nfe]
# path constrained properties to be monitored:
pc = ['Tad','T']
# monitored vars:
poi = [x for x in x_vars] + u
#parameter scenario:
scenario = {('A',('p',)):-0.2,('A',('i',)):-0.2,('kA',()):-0.2}           
            
# create MHE-NMPC-controller object
c = MHEGen(d_mod = SemiBatchPolymerization,
           y=y_vars,
           x=x_vars,           
           x_noisy=x_noisy,
           p_noisy=p_noisy,
           alpha = alpha,
           u=u,
           u_bounds = u_bounds,
           tf_bounds = tf_bounds,
           poi = x_vars,
           noisy_inputs = False,
           noisy_params = False,
           adapt_params = False,
           linapprox = True,
           update_uncertainty_set = False,
           process_noise_model = 'params_bias',
           obj_type='economic',
           nfe_t=nfe,
           ncp_t=ncp,
           path_constraints=pc)

# arguments for closed-loop simulation:
disturbance_src = {'disturbance_src':'parameter_scenario','scenario':scenario}
cov_matrices = {'y_cov':mcov,'q_cov':qcov,'u_cov':ucov,'p_cov':pcov}
reg_weights = {'K_w':1.0}
olrnmpc_args = {'cons':['mw','mw_ub','PO_ptg','unsat','temp_b','T_min','T_max']}
# run closed-loop simulation:
performance, iters = c.run(fix_noise=True,
                      advanced_step=False,
                      olrnmpc_args = olrnmpc_args,
                      disturbance_src=disturbance_src,
                      cov_matrices=cov_matrices,
                      regularization_weights=reg_weights,
                      meas_noise=x_measurement)

c.plant_simulation_model.check_feasibility(display=True)


""" visualization"""

#plot state trajectories and estimates
x = []
for i in range(1,iters+1):
    for cp in range(ncp+1):
        x.append(x[-cp-1]+c.pc_trajectory['tf',(i,cp)] if i > 1 else c.pc_trajectory['tf',(i,cp)])
x_e = [c.nmpc_trajectory[i,'tf'] for i in range(1,iters)]                
for var in poi[:-2]:
    if var == 'MX':
        for k in [0,1]:
            y_e = [c.nmpc_trajectory[i,(var,(k,))] for i in range(1,iters)]
            y = [c.monitor[i][var,(1,cp,k)] for i in range(1,iters+1) for cp in range(ncp+1)]    
            plt.figure(), plt.plot(x,y), plt.plot(x_e,y_e,'r',marker='x',linestyle='None'), plt.xlabel('time [min]'), plt.ylabel(var+str(k))
    else:
        y_e = [c.nmpc_trajectory[i,(var,())] for i in range(1,iters)]
        y = [c.monitor[i][var,(1,cp)] for i in range(1,iters+1) for cp in range(ncp+1)]
        plt.figure(), plt.plot(x,y), plt.plot(x_e,y_e,'r',marker='x',linestyle='None'), plt.xlabel('time [min]'), plt.ylabel(var)       

# path constraints
x = []
for i in range(1,iters+1):
    for cp in range(1,ncp+1):
        x.append(x[-cp]+c.pc_trajectory['tf',(i,cp)] if i > 1 else c.pc_trajectory['tf',(i,cp)])
y = [c.pc_trajectory['T',(i,(cp,))] for i in range(1,iters+1) for cp in range(1,ncp+1)]
plt.figure(), plt.plot(x,y,color='grey'), plt.plot([0,x[-1]],[423.15e-2,423.15e-2],'r--'), plt.plot([0,x[-1]],[373.15e-2,373.15e-2],'r--')
plt.xlabel('time [min]'), plt.ylabel('T')
y = [c.pc_trajectory['Tad',(i,(cp,))] for i in range(1,iters+1) for cp in range(1,ncp+1)]
plt.figure(), plt.plot(x,y,color='grey'), plt.plot([0,x[-1]],[443.15e-2,443.15e-2],'r--')
plt.xlabel('time [min]'), plt.ylabel('Tad')
    
#plot control profiles
x = [c.nmpc_trajectory[i,'tf'] for i in range(1,iters+1)]
for control in u:
    y = [c.nmpc_trajectory[i,control] for i in range(1,iters+1)]
    plt.figure(), plt.step(x,y),plt.step([0,x[0]],[y[0],y[0]],'C0'),plt.xlabel('time [min]'), plt.ylabel(control)

   
# visualize confidence region:
if c.update_uncertainty_set:    
    dimension = 3 # dimension n of the n x n matrix = #DoF
    rhs_confidence = chi2.isf(1.0-0.99,dimension) # 0.1**2*5% measurment noise, 95% confidence level, dimension degrees of freedo
    rows = {}
    
    # plot cube cube
    kA = np.array([0.8,1.2])#*e.nominal_parameter_values['kA',()] 
    Ap = np.array([0.8,1.2])#*e.nominal_parameter_values['A',('p',)]
    Ai = np.array([0.8,1.2])#*e.nominal_parameter_values['A',('i',)] 
    x = [Ap[0],Ap[1],Ap[0],Ap[1]]  
    y = [Ai[1],Ai[1],Ai[0],Ai[0]]
    X,Y = np.meshgrid(x,y)
    Z_u = np.array([[kA[1],kA[1],kA[1],kA[1]] for i in range(len(x))])
    Z_l = np.array([[kA[0],kA[0],kA[0],kA[0]] for i in range(len(x))])
    aux = {1:X,2:Y,3:(Z_l,Z_u)}
    combinations = [[1,2,3],[1,3,2],[3,1,2]]
    facets = {}
    b = 0
    for combination in combinations:
        facets[b] = np.array([aux[i] if i != 3  else aux[i][0] for i in combination])
        facets[b+1] = np.array([aux[i]  if i != 3  else aux[i][1] for i in combination]) 
        b += 2
        
    p_star = np.zeros(dimension)
    for key in scenario:
        p_star[c.PI_indices[key]]=(1+scenario[key])*c.nominal_parameter_values[key]
    p_star[2] *= c.olnmpc.Hrxn['p'].value
    
#    for facet in facets:
#        f = open('results/face'+str(facet)+'.txt','wb')
#        for i in range(4):
#            for j in range(4):
#                f.write(str(facets[facet][0][i][j]*c.nominal_parameter_values['A',('i',)]) + '\t' + str(facets[facet][1][i][j]*c.nominal_parameter_values['A',('p',)]) + '\t' + str(facets[facet][2][i][j]*c.nominal_parameter_values['kA',()]*c.olnmpc.Hrxn['p'].value) + '\n')
#            f.write('\n')
#        f.close()
        
    for r in range(1,7):
        A_dict = c.mhe_confidence_ellipsoids[r]
        center = np.zeros(dimension)
        for par in c.nmpc_trajectory[r,'e_pars']:
            center[c.PI_indices[par]] = c.nmpc_trajectory[r,'e_pars'][par]
        for m in range(dimension):
            rows[m] = np.array([A_dict[(m,i)] for i in range(dimension)])
        A = 1/rhs_confidence*np.array([np.array(rows[i]) for i in range(dimension)])
        U, s, V = np.linalg.svd(A) # singular value decomposition 
        radii = 1/np.sqrt(s) # length of half axes, V rotation
        
        # transform in polar coordinates for simpler waz of plotting
        u = np.linspace(0.0, 2.0 * np.pi, 30) # angle = idenpendent variable
        v = np.linspace(0.0, np.pi, 30) # angle = idenpendent variable
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) # x-coordinate
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) # y-coordinate
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        #f = open('results/data_ellipsoid'+str(r)+'.txt','wb')
        for i in range(len(x[0][:])):
            for j in range(len(x[:][0])):
                [x[i][j],y[i][j],z[i][j]] = np.dot(U,[x[i][j],y[i][j],z[i][j]]) + center
                z[i][j] *= c.olnmpc.Hrxn['p'].value
                #f.write(str(x[i][j]) + '\t' + str(y[i][j]) + '\t' + str(z[i][j]) + '\n')
            #f.write('\n')
        #f.close()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x,y,z,alpha = 0.1, edgecolor='r')
        ax.scatter(center[0],center[1],center[2]*c.olnmpc.Hrxn['p'].value,marker='o',color='r')       
        ax.scatter(p_star[0],p_star[1],p_star[2],marker='o',color='k')
        for i in facets:
            ax.plot_surface(facets[i][0]*c.nominal_parameter_values['A',('i',)],facets[i][1]*c.nominal_parameter_values['A',('p',)],facets[i][2]*c.nominal_parameter_values['kA',()]*c.olnmpc.Hrxn['p'].value,edgecolors='k',color='grey',alpha=0.1)
        scaling = np.array([0.5,1.5])
        ax.set_xlim(scaling*c.nominal_parameter_values['A',('i',)])
        ax.set_xlabel('\n' + r'$A_i$ [$\frac{m^3}{mol s}$]', linespacing=1.2)
        ax.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.set_xticks(np.array([2.5e5,4e5,5.5e5])*1e-4)
        ax.set_ylim(scaling*c.nominal_parameter_values['A',('p',)])
        ax.set_ylabel('\n' + r'$A_p$ [$\frac{m^3}{mol s}$]', linespacing=1.2)
        ax.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.set_yticks(np.array([8e3,14e3,20e3])*1e-4)
        ax.set_zlim(scaling*c.nominal_parameter_values['kA',()]*c.olnmpc.Hrxn['p'].value)
        ax.set_zlabel('\n' + r'$kA$ [$\frac{kJ}{K}$]', linespacing=1.2)
        ax.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.set_zticks(np.array([0.04*2,0.07*2,0.1*2])*c.olnmpc.Hrxn['p'].value)
        
        fig.tight_layout()
        #fig.savefig('results/125grid/'+str(r)+'.pdf')
        
        #ax.tick_params(axis='both',direction='in')
        
        #ax.view_init(15,35)
        # plot half axis
    plt.xlabel(r'$\Delta A_i$')
    plt.ylabel(r'$\Delta A_p$')    
    
# plot CPU times:
x = range(1,iters)
for k in ['olnmpc','lsmhe']:
    utime = [sum(performance[k,i][1][l].ru_utime-performance[k,i][0][l].ru_utime for l in [1]) for i in range(1,iters)]
    stime = [sum(performance[k,i][1][l].ru_stime-performance[k,i][0][l].ru_stime for l in [1]) for i in range(1,iters)]
    plt.figure(), plt.title(k+' - required CPU time')
    plt.bar(x,utime,label='utime'), plt.bar(x,stime,bottom=utime,color='C1',label='stime')
    plt.ylabel(r'$t_{CPU} [s]$'), plt.xlabel('iteration'), plt.legend()
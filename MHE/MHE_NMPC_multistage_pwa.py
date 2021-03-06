#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:51:51 2017
@author: flemmingholtorf
"""
#### 
from __future__ import print_function
from pyomo.environ import *
from main.dync.MHEGen_multistage import MheGen
from main.mods.final_pwa.mod_class_cj_pwa_multistage import *
from main.mods.final_pwa.mod_class_cj_pwa import *
import sys
import itertools, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import chi2
from main.noise_characteristics_cj import *
from mpl_toolkits.mplot3d import Axes3D

# redirect system output to a file:
#sys.stdout = open('consol_output','w')

###############################################################################
###                               Specifications
###############################################################################

states = ["PO","MX","MY","Y","W","PO_fed","T","T_cw"]
x_noisy = ["PO","MX","MY","Y","W","T"] 
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)],"T":[()],"T_cw":[()]}
p_noisy = {"A":[('p',),('i',)],'kA':[()]}
u = ["u1", "u2"]
u_bounds = {"u1": (-5.0, 5.0), "u2": (0.0, 3.0)} 

#y = {"Y","PO", "W", "MY", "MX", "MW","m_tot",'T'}
#y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()],'T':[()]}
y = {"PO", 'T', 'MY', 'Y'}
y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()],'T':[()]}
nfe = 24
tf_bounds = [10.0*24.0/nfe, 30.0*24.0/nfe]

pc = ['Tad','T']
scenario = {('A',('p',)):-0.1,('A',('i',)):-0.1,('kA',()):-0.1}
# scenario_tree
st = {} # scenario tree : {parent_node, scenario_number on current stage, base node (True/False), scenario values {'name',(index):value}}
s_max = 9
nr = 1
alpha = 0.2
for i in range(1,nfe+1):
    if i < nr + 1:
        for s in range(1,s_max**i+1):
            if s%s_max == 1:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),True,{('A',('p',)):1.0,('A',('i',)):1.0,('kA',()):1.0}) 
            elif s%s_max == 2:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
            elif s%s_max == 3:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
            elif s%s_max == 4:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
            elif s%s_max == 5:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
            elif s%s_max == 6:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
            elif s%s_max == 7:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
            elif s%s_max == 8:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
            else:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
    else:
        for s in range(1,s_max**nr+1):
            st[(i,s)] = (i-1,s,True,st[(i-1,s)][3])

sr = s_max**nr


e = MheGen(d_mod=SemiBatchPolymerization_multistage,
           d_mod_mhe = SemiBatchPolymerization,
           y=y,
           y_vars=y_vars,
           x_noisy=x_noisy,
           x_vars=x_vars,
           p_noisy=p_noisy,
           states=states,
           u=u,
           u_bounds = u_bounds,
           tf_bounds = tf_bounds,
           scenario_tree = st,
           robust_horizon = nr,
           s_max = sr,
           noisy_inputs = False,
           noisy_params = True,
           adapt_params = True,
           update_scenario_tree = True,
           process_noise_model = None,#'params_bias',
           confidence_threshold = alpha,
           robustness_threshold = 0.05,
           estimate_exceptance = 10000,

           obj_type='tracking',
           nfe_t=nfe,
           sens=None,
           diag_QR=True,
           del_ics=False,
           path_constraints=pc)
###############################################################################
###                                     NMPC
###############################################################################
e.recipe_optimization()
e.set_reference_state_trajectory(e.get_state_trajectory(e.recipe_optimization_model))
e.set_reference_control_trajectory(e.get_control_trajectory(e.recipe_optimization_model))
e.generate_state_index_dictionary()

e.create_enmpc()
k = 1 
for i in range(1,nfe):
    print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
    e.create_mhe()
    if i == 1:
        #e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call = True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.plant_simulation(e.store_results(e.recipe_optimization_model),first_call=True,disturbance_src = "parameter_scenario",scenario=scenario)
        e.set_measurement_prediction(e.store_results(e.recipe_optimization_model))
        e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
        e.cycle_mhe(e.store_results(e.recipe_optimization_model),mcov,qcov,ucov,p_cov=pcov,first_call=True) 
        e.cycle_nmpc(e.store_results(e.recipe_optimization_model))
    else:
        #e.plant_simulation(e.store_results(e.olnmpc),disturbance_src="parameter_noise",parameter_disturbance=v_param)
        e.plant_simulation(e.store_results(e.olnmpc),disturbance_src = "parameter_scenario",scenario=scenario)
        e.set_measurement_prediction(e.store_results(e.forward_simulation_model))
        e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  
        e.cycle_mhe(previous_mhe,mcov,qcov,ucov,p_cov=pcov) 
        e.cycle_nmpc(e.store_results(e.olnmpc))  
    
    # here measurement becomes available
    e.load_reference_trajectories()
    previous_mhe = e.solve_mhe(fix_noise=True) # solves the mhe problem
#    if i == 2:
#        sys.exit()
    if e.update_scenario_tree:
        e.compute_confidence_ellipsoid()

#    if i ==2:
#        sys.exit()
    # solve the advanced step problems
    e.cycle_ics_mhe(nmpc_as=False,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc
    e.solve_olnmpc() # solves the olnmpc problem

    
    e.olnmpc.write_nl()
    e.cycle_iterations()
    k += 1
   
    #troubleshooting
    if  e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.plant_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.nmpc_trajectory[i,'solstat_mhe'] != ['ok','optimal']:
        break

# simulate the last step too

for i in range(1,k):
    print('iteration: %i' % i)
    print('open-loop optimal control: ', end='')
    print(e.nmpc_trajectory[i,'solstat'],e.nmpc_trajectory[i,'obj_value'])
    print('constraint inf: ', e.nmpc_trajectory[i,'eps'])
    print('plant: ',end='')
    print(e.plant_trajectory[i,'solstat'])
    
e.plant_simulation(e.store_results(e.olnmpc))

# uncertainty realization
print('uncertainty realization')
for p in p_noisy:
    pvar_r = getattr(e.plant_simulation_model, p)
    pvar_m = getattr(e.recipe_optimization_model, p)
    for key in p_noisy[p]:
        if key != ():
            print('delta_p ',p,key,': ',(pvar_r[key].value-pvar_m[key].value)/pvar_m[key].value)
        else:
            print('delta_p ',p,key,': ',(pvar_r.value-pvar_m.value)/pvar_m.value)


###############################################################################
####                        plot results comparisons   
###############################################################################

t_traj_nmpc = np.array([e.nmpc_trajectory[i,'tf'] for i in range(1,k)])
t_traj_sim = np.array([e.nmpc_trajectory[i,'tf'] for i in range(1,k+1)])
plt.figure(1)

t = e.get_tf(1)
l = 0

#moments
moment = ['MX']
for m in moment:
    for j in range(0,2):
        state_traj_ref = np.array([e.reference_state_trajectory[(m,(i,3,j,1))] for i in range(1,nfe+1)]) 
        state_traj_nmpc = np.array([e.nmpc_trajectory[i,(m,(j,))] for i in range(1,k)])
        state_traj_sim = np.array([e.plant_trajectory[i,(m,(j,))] for i in range(1,k+1)])
        plt.figure(l)
        plt.plot(t,state_traj_ref, label = "reference")
        plt.plot(t_traj_nmpc,state_traj_nmpc, label = "mhe/nmpc")
        plt.plot(t_traj_sim,state_traj_sim, label = "plant")
        plt.ylabel(m+str(j))
        plt.legend()
        l += 1

plots = [('T',()),('T_cw',()),('Y',()),('PO',()),('PO_fed',()),('W',()),('MY',())]
for p in plots: 
    state_traj_ref = np.array([e.reference_state_trajectory[(p[0],(i,3,1))] for i in range(1,nfe+1)]) 
    state_traj_nmpc = np.array([e.nmpc_trajectory[i,p] for i in range(1,k)])
    state_traj_sim = np.array([e.plant_trajectory[i,p] for i in range(1,k+1)])    
    plt.figure(l)
    plt.plot(t,state_traj_ref, label = "reference")
    plt.plot(t_traj_nmpc,state_traj_nmpc, label = "mhe/nmpc")
    plt.plot(t_traj_sim,state_traj_sim, label = "plant")
    plt.legend()
    plt.ylabel(p[0])
    l += 1
    
plots = ['u1','u2']
t_traj_nmpc = [e.nmpc_trajectory[i,'tf'] for i in range(0,k+1)]
aux1 = []
for i in range(1,len(t_traj_nmpc)):
    aux1.append(t_traj_nmpc[i-1])
    aux1.append(t_traj_nmpc[i])

t_traj_nmpc = np.array(aux1)
t_traj_sim = t_traj_nmpc 
for b in plots:
    aux_ref = {}
    for s in range(1,s_max+1):
        aux_ref[s] = []
    aux_nmpc = []
    aux_sim = []
    for i in range(1,k+1):
        for z in range(2):
            for s in range(1,s_max+1):
                aux_ref[s].append(e.reference_control_trajectory[b,(i,s)]) # reference computed off-line/recipe optimization
            aux_nmpc.append(e.nmpc_trajectory[i,b]) # nmpc --> predicted one step ahead
            aux_sim.append(e.plant_trajectory[i,b]) # after advanced step --> one step ahead
    
    control_traj_nmpc = np.array(aux_nmpc)
    control_traj_sim = np.array(aux_sim)
    plt.figure(l)
    for s in range(1,s_max+1):
        t = [0] + e.get_tf(s)
        aux2 = []
        for i in range(1,k+1):
            aux2.append(t[i-1])
            aux2.append(t[i])
        t = np.array(aux2)
        control_traj_ref = np.array(aux_ref[s])
        plt.plot(t,control_traj_ref, color='blue',label = "reference scenario" + str(s))
    plt.plot(t_traj_nmpc,control_traj_nmpc, label = "predicted")
    plt.plot(t_traj_sim,control_traj_sim, label = "SBU")
    plt.legend()
    plt.ylabel(b)
    l += 1
    
e.plant_simulation_model.check_feasibility(display=True)







###############################################################################
###         Plotting path constraints
###############################################################################

l += 1
heat_removal = {}
t = {}
Tad = {}
path_constraints = {}
path_constraints[0] = e.pc_trajectory
for i in range(1): # loop over all runs
    if path_constraints[i] =='error':
        continue
    heat_removal[i] = []
    t[i] = []
    Tad[i] = []
    for fe in range(1,25):
        for cp in range(1,4):        
            heat_removal[i].append(path_constraints[i]['T',(fe,(cp,))])
            Tad[i].append(path_constraints[i]['Tad',(fe,(cp,))])
            if fe > 1:
                t[i].append(t[i][-cp]+path_constraints[i]['tf',(fe,cp)])
            else:
                t[i].append(path_constraints[i]['tf',(fe,cp)])
    
    
max_tf = max(t[0])   
plt.figure(l)
for i in Tad:
    plt.plot(t[i],Tad[i], color='grey')
plt.plot([0,max_tf],[4.4315,4.4315], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('T [K]')
    
l += 1
plt.figure(l)
for i in heat_removal:
    plt.plot(t[i],heat_removal[i], color='grey')
plt.plot([0,max_tf],[4.4315,4.4315], color='red', linestyle='dashed')
plt.plot([0,max_tf],[3.7315,3.7315], color='red', linestyle='dashed')
plt.xlabel('t [min]')
plt.ylabel('T [k]')
l += 1 

###############################################################################
###         Plotting 1st Order Approximation of Confidence Region 
###############################################################################
try:
    #fig = plt.figure(l)
    #ax = fig.add_subplot(111, projection='3d')
    
    dimension = 3 # dimension n of the n x n matrix = #DoF
    rhs_confidence = chi2.isf(1.0-0.99,dimension) # 0.1**2*5% measurment noise, 95% confidence level, dimension degrees of freedo
    rows = {}
    
    # cube
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
        p_star[e.PI_indices[key]]=(1+scenario[key])*e.nominal_parameter_values[key]
    p_star[2] *= e.olnmpc.Hrxn['p'].value
    
   
    for facet in facets:
        f = open('results/face'+str(facet)+'.txt','wb')
        for i in range(4):
            for j in range(4):
                f.write(str(facets[facet][0][i][j]*e.nominal_parameter_values['A',('i',)]) + '\t' + str(facets[facet][1][i][j]*e.nominal_parameter_values['A',('p',)]) + '\t' + str(facets[facet][2][i][j]*e.nominal_parameter_values['kA',()]*e.olnmpc.Hrxn['p'].value) + '\n')
            f.write('\n')
        f.close()
    for r in range(1,7):
        A_dict = e.mhe_confidence_ellipsoids[r]
        center = np.zeros(dimension)
        for par in e.nmpc_trajectory[r,'e_pars']:
            center[e.PI_indices[par]] = e.nmpc_trajectory[r,'e_pars'][par]
        for m in range(dimension):
            rows[m] = np.array([A_dict[(m,i)] for i in range(dimension)])
        A = 1/rhs_confidence*np.array([np.array(rows[i]) for i in range(dimension)])
        U, s, V = linalg.svd(A) # singular value decomposition 
        radii = 1/np.sqrt(s) # length of half axes, V rotation
        
        # transform in polar coordinates for simpler waz of plotting
        u = np.linspace(0.0, 2.0 * np.pi, 30) # angle = idenpendent variable
        v = np.linspace(0.0, np.pi, 30) # angle = idenpendent variable
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) # x-coordinate
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) # y-coordinate
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        f = open('results/data_ellipsoid'+str(r)+'.txt','wb')
        for i in range(len(x[0][:])):
            for j in range(len(x[:][0])):
                [x[i][j],y[i][j],z[i][j]] = np.dot(U,[x[i][j],y[i][j],z[i][j]]) + center
                z[i][j] *= e.olnmpc.Hrxn['p'].value
                f.write(str(x[i][j]) + '\t' + str(y[i][j]) + '\t' + str(z[i][j]) + '\n')
            f.write('\n')
        f.close()
        #plt.plot(x,y, label = str(r))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x,y,z,alpha = 0.1, edgecolor='r')
        ax.scatter(center[0],center[1],center[2]*e.olnmpc.Hrxn['p'].value,marker='o',color='r')       
        ax.scatter(p_star[0],p_star[1],p_star[2],marker='o',color='k')
        for i in facets:
            ax.plot_surface(facets[i][0]*e.nominal_parameter_values['A',('i',)],facets[i][1]*e.nominal_parameter_values['A',('p',)],facets[i][2]*e.nominal_parameter_values['kA',()]*e.olnmpc.Hrxn['p'].value,edgecolors='k',color='grey',alpha=0.1)
        scaling = np.array([0.5,1.5])
        ax.set_xlim(scaling*e.nominal_parameter_values['A',('i',)])
        ax.set_xlabel('\n' + r'$A_i$ [$\frac{m^3}{mol s}$]', linespacing=1.2)
        ax.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.set_xticks([2.5e5,4e5,5.5e5])
        ax.set_ylim(scaling*e.nominal_parameter_values['A',('p',)])
        ax.set_ylabel('\n' + r'$A_p$ [$\frac{m^3}{mol s}$]', linespacing=1.2)
        ax.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.set_yticks([8e3,14e3,20e3])
        ax.set_zlim(scaling*e.nominal_parameter_values['kA',()]*e.olnmpc.Hrxn['p'].value)
        ax.set_zlabel('\n' + r'$kA$ [$\frac{kJ}{K}$]', linespacing=1.2)
        ax.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))
        ax.set_zticks(np.array([0.04,0.07,0.1])*e.olnmpc.Hrxn['p'].value)
        
        fig.tight_layout()
        fig.savefig('results/125grid/'+str(r)+'.pdf')
        
        #ax.tick_params(axis='both',direction='in')
        
        #ax.view_init(15,35)
        # plot half axis
    plt.xlabel(r'$\Delta A_i$')
    plt.ylabel(r'$\Delta A_p$')
except KeyError:
    pass

print('MULTISTAGE NMPC')
print('OPTIONS:')
print('measured vars ', e.y)
print('state vars ', e.states)
print('pars estimated online ', e.noisy_params)
print('pars adapted ', e.adapt_params)
print('update ST ', e.update_scenario_tree)
print('confidence threshold ', e.confidence_threshold)
print('robustness threshold ', e.robustness_threshold)
print('estimate_acceptance ', e.estimate_acceptance)

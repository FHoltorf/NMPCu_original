#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:51:51 2017

@author: flemmingholtorf
"""
#### 

from __future__ import print_function
from pyomo.environ import *
# from nmpc_mhe.dync.MHEGen import MheGen
from main.dync.MHEGen_multistage import MheGen
from main.mods.mod_class import *
from main.mods.mod_class_multistage import *
import sys
import itertools, sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import chi2
from main.noise_characteristics import *

# redirect system output to a file:
#sys.stdout = open('consol_output','w')

###############################################################################
###                               Specifications
###############################################################################

states = ["PO","MX","MY","Y","W","PO_fed"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
x_noisy = ["PO","MX","MY","Y","W","PO_fed"] # all the states are noisy  
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)]}
p_noisy = {"A":['p','i']}
u = ["u1", "u2"]
u_bounds = {"u1": (373.15/1e2, 443.15/1e2), "u2": (0, 3.0)} # 14.5645661157

# measured variables
#y = ["heat_removal","m_tot","MW","PO"] 
#y_vars = {"heat_removal":[()],"m_tot":[()],"MW":[()],"PO":[()]}
y = {"PO", "Y", "W", "MY", "MX", "MW","m_tot"}
y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()]}


pc = ['Tad','heat_removal']

# scenario_tree
st = {}
s_max = 5
nr = 1
nfe = 24
alpha = 0.1
for i in range(1,nfe+1):
    if i < nr + 1:
        for s in range(1,s_max**i+1):
            if s%s_max == 1:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),True,{'p':1.0,'i':1.0})
            elif s%s_max == 2:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{'p':1.0+alpha,'i':1.0+alpha})
            elif s%s_max == 3:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{'p':1.0-alpha,'i':1.0+alpha})
            elif s%s_max == 4:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{'p':1.0+alpha,'i':1.0-alpha})
            else:
                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{'p':1.0-alpha,'i':1.0-alpha})
    else:
        for s in range(1,s_max**nr+1):
            st[(i,s)] = (i-1,s,True,st[(i-1,s)][3])

sr = s_max**nr

e = MheGen(d_mod=SemiBatchPolymerization_multistage,
           d_mod_mhe=SemiBatchPolymerization,
           y=y,
           x_noisy=x_noisy,
           y_vars=y_vars,
           x_vars=x_vars,
           p_noisy=p_noisy,
           states=states,
           u=u,
           scenario_tree = st,
           robust_horizon = nr,
           s_max = sr,
           noisy_inputs = False,
           noisy_params = True,
           adapt_params = True,
           update_scenario_tree = False,
           confidence_threshold = 0.2,
           robustness_threshold = 0.05,
           obj_type='economic',
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
e.create_enmpc() # with tracking-type regularization

e.create_mhe()

k = 1 


before = {}
after = {}
diff = {}
applied = {}
state_offset = {} 
curr_estate = {}
curr_pstate = {}
 
for i in range(1,nfe):
    print('#'*21 + '\n' + ' ' * 10 + str(i) + '\n' + '#'*21)
    e.create_mhe()
    nfe_new = nfe - i
    if i == 1:
        e.plant_simulation(e.store_results(e.recipe_optimization_model),disturbances=v_disturbances,first_call=True,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.set_prediction(e.store_results(e.recipe_optimization_model)) # only required for asMHE
        e.cycle_mhe(e.store_results(e.recipe_optimization_model),mcov,qcov,ucov, first_call=True) #adjusts the mhe problem according to new available measurements
        e.cycle_nmpc(e.store_results(e.recipe_optimization_model),nfe_new)
    else:
        e.plant_simulation(e.store_results(e.olnmpc),disturbances=v_disturbances,disturbance_src = "parameter_noise",parameter_disturbance = v_param)
        e.set_prediction(e.store_results(e.forward_simulation_model))
        e.cycle_mhe(previous_mhe,mcov,qcov,ucov) # only required for asMHE
        e.cycle_nmpc(e.store_results(e.olnmpc),nfe_new)   

    # solve the advanced step problems
    e.cycle_ics_mhe(nmpc_as=True,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc

    e.solve_olnmpc() # solves the olnmpc problem

    e.create_suffixes_nmpc()
    e.sens_k_aug_nmpc()
    
    e.create_measurement(e.store_results(e.plant_simulation_model),x_measurement)  

    # solve mhe problem
    e.solve_mhe(fix_noise=True) # solves the mhe problem
    previous_mhe = e.store_results(e.lsmhe)
    e.compute_confidence_ellipsoid()

    # update state estimate 
    e.update_state_mhe() # can compute offset within this function by setting as_nmpc_mhe_strategy = True
    
    # compute fast update for nmpc
    state_offset[i], curr_pstate[i], curr_estate[i] = e.compute_offset_state(src_kind="estimated")
    before[i], after[i], diff[i], applied[i] = e.sens_dot_nmpc()   

    #sIpopt
    #e.create_nmpc_sIpopt_suffixes()
    #e.nmpc_sIpopt_update()

#    if i == 1:
#        break
    # forward simulation for next iteration
    e.forward_simulation()
    
    e.cycle_iterations()
    k += 1
   

    #troubleshooting
    if  e.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.nmpc_trajectory[i,'solstat_mhe'] != ['ok','optimal'] or \
        e.plant_trajectory[i,'solstat'] != ['ok','optimal'] or \
        e.simulation_trajectory[i,'solstat'] != ['ok','optimal']:
        with open("000aaa.txt","w") as f:
            f.write('plant :' + e.plant_trajectory[i,'solstat'][1] + '\n' \
                    + 'nmpc :' + e.nmpc_trajectory[i,'solstat'][1] + '\n' \
                    + 'simulation :' + e.simulation_trajectory[i,'solstat'][1])
        break

# simulate the last step too

#e.forward_simulation_model.troubleshooting()
e.plant_simulation_model.troubleshooting()

for i in range(1,k):
    print('iteration: %i' % i)
    print('open-loop optimal control: ', end='')
    print(e.nmpc_trajectory[i,'solstat'],e.nmpc_trajectory[i,'obj_value'])
    print('constraint inf: ', e.nmpc_trajectory[i,'eps'])
    print('mhe: ', end='')
    print(e.nmpc_trajectory[i,'solstat_mhe'], e.nmpc_trajectory[i,'obj_value_mhe'])
    print('plant: ',end='')
    print(e.plant_trajectory[i,'solstat'])
#    print('forward_simulation: ',end='')
#    print(e.simulation_trajectory[i,'solstat'], e.simulation_trajectory[i,'obj_fun'])


e.plant_simulation(e.store_results(e.olnmpc))


###############################################################################
####                        plot results comparisons   
###############################################################################

t_traj_nmpc = np.array([e.nmpc_trajectory[i,'tf'] for i in range(1,k)])
t_traj_sim = np.array([e.nmpc_trajectory[i,'tf'] for i in range(1,k+1)])
plt.figure(1)

t = e.get_tf()
#
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

plots = [('Y',()),('PO',()),('PO_fed',()),('W',()),('MY',())]
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
t = [0] + t
aux1 = []
aux2 = []
for i in range(1,len(t_traj_nmpc)):
    aux1.append(t_traj_nmpc[i-1])
    aux1.append(t_traj_nmpc[i])
    aux2.append(t[i-1])
    aux2.append(t[i])
    
t = np.array(aux2)
t_traj_nmpc = np.array(aux1)
t_traj_sim = t_traj_nmpc 
for b in plots:
    aux_ref = []
    aux_nmpc = []
    aux_sim = []
    for i in range(1,k+1):
        for z in range(2):
            aux_ref.append(e.reference_control_trajectory[b,(i,1)]) # reference computed off-line/recipe optimization
            aux_nmpc.append(e.nmpc_trajectory[i,b]) # nmpc --> predicted one step ahead
            aux_sim.append(e.plant_trajectory[i,b]) # after advanced step --> one step ahead
    control_traj_ref = np.array(aux_ref)
    control_traj_nmpc = np.array(aux_nmpc)
    control_traj_sim = np.array(aux_sim)
    plt.figure(l)
    plt.plot(t,control_traj_ref, label = "reference")
    plt.plot(t_traj_nmpc,control_traj_nmpc, label = "predicted")
    plt.plot(t_traj_sim,control_traj_sim, label = "SBU")
    plt.legend()
    plt.ylabel(b)
    l += 1
    
e.plant_simulation_model.check_feasibility(display=True)



t = np.array(aux2)
t_traj_nmpc = np.array(aux1)
t_traj_sim = t_traj_nmpc 
for b in plots:
    aux_ref = []
    aux_nmpc = []
    aux_sim = []
    for i in range(1,k+1):
        for z in range(2):
            aux_ref.append(e.reference_control_trajectory[b,(i,1)])
            aux_nmpc.append(e.nmpc_trajectory[i,b])
            aux_sim.append(e.plant_trajectory[i,b])
    control_traj_ref = np.array(aux_ref)
    control_traj_nmpc = np.array(aux_nmpc)
    control_traj_sim = np.array(aux_sim)
    plt.figure(l)
    plt.plot(t,control_traj_ref, label = "reference")
    plt.plot(t_traj_nmpc,control_traj_nmpc, label = "predicted")
    plt.plot(t_traj_sim,control_traj_sim, label = "SBU")
    plt.legend()
    plt.ylabel(b)
    l += 1
    
e.plant_simulation_model.check_feasibility(display=True)

state_offset_norm = []
diff_norm = []

for u in ['u1','u2']:    
    x = []
    for i in range(1,k):
        x.append(diff[i][u])
    l += 1
    plt.figure(l)
    plt.plot(x)

for p in [('Y',(1,)),('PO',(1,)),('PO_fed',(1,)),('W',(1,))]:
    x = []
    y = [] 
    z = []
    for i in range(1,k):
        #x.append(curr_pstate[i][p])
        #y.append(curr_estate[i][p])
        #z.append(curr_pstate[i][p] - state_offset[i][p])
        z.append(state_offset[i][p])
    l += 1
    plt.figure(l)
    #plt.plot(x, label = 'predicted')
    #plt.plot(y, label = 'estimated')
    plt.plot(z, label = 'estimated check')
    plt.ylabel(p[0])
    plt.legend()
    

# compute the confidence ellipsoids
# delta_theta^T*Vi*delta_theta = sigma^2*chi2(n_dof,confidence_level)
l += 1
plt.figure(l)

###############################################################################
###         Plotting 1st Order Approximation of Confidence Region 
###############################################################################
dimension = 2 # dimension n of the n x n matrix = #DoF
rhs_confidence = chi2.isf(1.0-0.95,dimension) # 0.1**2*5% measurment noise, 95% confidence level, dimension degrees of freedo
rows = {}
for r in range(1,k):
    A_dict = e.mhe_confidence_ellipsoids[r]
    center = [0,0]
    for m in range(dimension):
        rows[m] = np.array([A_dict[(m,i)] for i in range(dimension)])
    A = 1/rhs_confidence*np.array([np.array(rows[i]) for i in range(dimension)])
    center = np.array([0]*dimension)
    U, s, V = linalg.svd(A) # singular value decomposition 
    radii = 1/np.sqrt(s) # length of half axes, V rotation
    
    # transform in polar coordinates for simpler waz of plotting
    theta = np.linspace(0.0, 2.0 * np.pi, 100) # angle = idenpendent variable
    x = radii[0] * np.sin(theta) # x-coordinate
    y = radii[1] * np.cos(theta) # y-coordinate
    for i in range(len(x)):
        [x[i],y[i]] = np.dot([x[i],y[i]], V) + center
    plt.plot(x,y, label = str(r))

    
    # plot half axis
for p in range(dimension):
    x = radii[p]*U[p][0]
    y = radii[p]*U[p][1]
    plt.plot([0,x],[0,y],color='red')
plt.xlabel(r'$\Delta A_i$')
plt.ylabel(r'$\Delta A_p$')
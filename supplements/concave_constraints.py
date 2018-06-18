#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:51:51 2017
@author: flemmingholtorf
"""

from __future__ import print_function
from pyomo.environ import *
import sys
import itertools, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
from pyomo.environ import *
from scipy.stats import chi2
from copy import deepcopy
from main.dync.MHEGen_adjusted import MheGen
from main.mods.final_pwa.mod_class_cj_pwa import * #_robust_optimal_control
from main.noise_characteristics_cj import * 
# monitor CPU time
#CPU_t = {}
#
#states = ["PO","MX","MY","Y","W","PO_fed","T","T_cw"] # ask about PO_fed ... not really a relevant state, only in mathematical sense
#x_noisy = ["PO","MX","MY","Y","W","T"] # all the states are noisy  
#x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)], "T":[()], "T_cw":[()]}
#
#u = ["u1", "u2"]
#u_bounds = {"u1": (-5.0, 5.0), "u2": (0.0, 3.0)} 
#
##    y = {"Y","PO", "W", "MY", "MX", "MW","m_tot",'T'}
##    y_vars = {"Y":[()],"PO":[()],"MW":[()], "m_tot":[()],"W":[()],"MX":[(0,),(1,)],"MY":[()],'T':[()]}
#y = {"MY","Y","PO",'T'}#"m_tot"
#y_vars = {"MY":[()],"Y":[()],"PO":[()],'T':[()]} #"m_tot":[()],,"MW":[()]}
#
#nfe = 24
#tf_bounds = [10.0*24.0/nfe, 30.0*24.0/nfe]
#
#cons = ['mw','mw_ub','PO_ptg','unsat','temp_b','T_min','T_max']
#pc = ['Tad','T']
#p_noisy = {"A":[('p',),('i',)],'kA':[()]}
#alpha = {('A',('p',)):0.2,('A',('i',)):0.2,('kA',()):0.2}
##         ('PO_ic',()):0.02,('T_ic',()):0.005,
##         ('MY_ic',()):0.01,('MX_ic',(1,)):0.005}
#e = MheGen(d_mod=SemiBatchPolymerization,
#           linapprox = True,
#           alpha = alpha,
#           x_noisy=x_noisy,
#           x_vars=x_vars,
#           y=y,
#           y_vars=y_vars,
#           states=states,
#           p_noisy=p_noisy,
#           u=u,
#           noisy_inputs = False,
#           noisy_params = True,
#           adapt_params = True,
#           update_uncertainty_set = True,
#           process_noise_model = None,#'params_bias',
#           u_bounds=u_bounds,
#           tf_bounds = tf_bounds,
#           diag_QR=False,
#           nfe_t=nfe,
#           del_ics=False,
#           sens=None,
#           obj_type='tracking',
#           path_constraints=pc)
#e.delta_u = True

###############################################################################
###                                     NMPC
###############################################################################
#e.recipe_optimization(cons=cons,eps=1e-4)

#mod = e.recipe_optimization_model
mod = SemiBatchPolymerization(24,3)
mod.initialize_element_by_element()
mod.create_bounds()
mod.clear_aux_bounds()
ip = SolverFactory('ipopt')
ip.solve(mod, tee=True)
mod.u1.fix()
mod.u2.fix()
mod.tf.fix()
constraints = {}
nom_value1 = mod.A['p'].value
nom_value2 = mod.A['i'].value
nom_value3 = mod.kA.value
for i in range(21):
    mod.A['p'].value = nom_value1*(0.8 + 0.4 * i/20)
#    mod.A['i'].value = nom_value2*(0.8 + 0.4 * i/20)
#    mod.kA.value = nom_value3*(1.2 + 0.4 * i/20)
    ip.solve(mod, tee=True)
#    constraints[('unsat',i)] = mod.unsat_value*mod.m_tot[24,3,1].value*mod.m_tot_scale - 1000.0*(mod.MY[24,3,1].value*mod.MY0_scale + mod.Y[24,3,1].value*mod.Y_scale) + mod.eps[2,1].value#- mod.check_feasibility()['epc_unsat'] + mod.unsat_value.value
#    constraints[('unreac',i)] = mod.unreacted_PO*1e-6*mod.m_tot[24,3,1].value*mod.m_tot_scale - mod.PO[24,3,1].value*mod.PO_scale*mod.mw_PO + mod.eps[1,1].value #- mod.check_feasibility()['epc_PO_ptg'] + mod.unreacted_PO.value
#    constraints[('mw',i)] = mod.MX[24,3,1,1].value*mod.MX1_scale - (mod.molecular_weight.value - mod.mw_PG.value)/mod.mw_PO.value/mod.num_OH*mod.MX[24,3,0,1].value*mod.MX0_scale + mod.eps[3,1].value # mod.check_feasibility()['epc_mw'] + mod.molecular_weight.value
    constraints[('unsat',i)] = mod.unsat_value*mod.m_tot[24,3].value*mod.m_tot_scale - 1000.0*(mod.MY[24,3].value*mod.MY0_scale + mod.Y[24,3].value*mod.Y_scale) #+ mod.eps[2].value#- mod.check_feasibility()['epc_unsat'] + mod.unsat_value.value
    constraints[('unreac',i)] = mod.unreacted_PO*1e-6*mod.m_tot[24,3].value*mod.m_tot_scale - mod.PO[24,3].value*mod.PO_scale*mod.mw_PO #+ mod.eps[1].value #- mod.check_feasibility()['epc_PO_ptg'] + mod.unreacted_PO.value
    constraints[('mw',i)] = mod.MX[24,3,1].value*mod.MX1_scale - (mod.molecular_weight.value - mod.mw_PG.value)/mod.mw_PO.value/mod.num_OH*mod.MX[24,3,0].value*mod.MX0_scale #+ mod.eps[3].value # mod.check_feasibility()['epc_mw'] + mod.molecular_weight.value
    for j in range(1,25):
        for cp in range(1,4):
            constraints[('T',i,j,cp)] = mod.T[j,cp].value 
            constraints[('Tad',i,j,cp)] = mod.Tad[j,cp].value
        
    
plt.figure(0)
x = np.array([0.8+ 0.4*i/20.0 for i in range(21)])
y = np.array([constraints[('unsat',i)] for i in range(21)])
plt.plot(x,y)

plt.figure(1)
y = np.array([constraints[('unreac',i)] for i in range(21)])
plt.plot(x,y)

plt.figure(2)
y = np.array([constraints[('mw',i)] for i in range(21)])
plt.plot(x,y)

y = [(j + mod.tau_i_t[cp])/24.0 for j in range(0,24) for cp in range(1,4)]
X,Y = np.meshgrid(x,y)
Z1 = [[constraints[('T',i,j,cp)] for i in range(21)] for j in range(1,25) for cp in range(1,4)]
Z2 = [[constraints[('Tad',i,j,cp)] for i in range(21)] for j in range(1,25) for cp in range(1,4)]
            
fig=plt.figure(3)
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z1,color='b',alpha = 0.5)
plt.show()


fig=plt.figure(4)
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z2,color='b',alpha = 0.5)
plt.show()

plt.figure(5)
for j in range(1,25):
    for cp in range(1,4):
        y = np.array([constraints[('T',i,j,cp)] for i in range(21)])
        plt.plot(x,y)

plt.figure(6)
for j in range(1,25):
    for cp in range(1,4):
        y = np.array([constraints[('Tad',i,j,cp)] for i in range(21)])
        plt.plot(x,y)
#y = np.array([constraints[('T',i)] for i in range(21)])
#plt.plot(x,y)

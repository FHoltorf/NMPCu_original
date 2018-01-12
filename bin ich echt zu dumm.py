#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 22:27:36 2018

@author: flemmingholtorf
"""
from pyomo.core import *
from pyomo.environ import *
import numpy as np
from pyomo.dae import *
from pyomo.opt import ProblemFormat
import collections
import matplotlib.pyplot as plt
from six import itervalues, iterkeys, iteritems
import sys
from scipy.stats import chi2

for N in range(5,10):
    m = ConcreteModel()
    m.k = Set(initialize=[i for i in range(N)])
    
    m.y = Var(m.k, initialize = 0)
    m.u = Param(m.k, mutable=True)
    m.pi1 = Var(initialize = 1)
    m.pi2 = Var(initialize = 1)
    #m.pi3 = Var(initialize = 1)
    
    m.y_m = Param(m.k, mutable = True)
    u_nom = np.linspace(1,N,N)
    nom = 23*u_nom-0.5*u_nom**2#+0.5*u_nom**3
    print(nom)
    for i in range(N):    
        nom[i] = np.random.normal(loc=nom[i], scale=1.0)
        m.y_m[i] = nom[i]
        m.u[i] = u_nom[i]
        
    def rule(m,k):
        return m.y[k] - m.pi1*m.u[k]-m.pi2*m.u[k]**2 == 0.0 #+m.pi3*m.u[k]**3 
    
    m.const = Constraint(m.k, rule=rule)
    
    m.obj = Objective(rule=lambda m: 0.5*sum((m.y[i] - m.y_m[i])**2 for i in m.k))
    m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)      
    m.dof_v = Suffix(direction=Suffix.EXPORT) 
    
    m.pi1.set_suffix_value(m.dof_v, 1)
    m.pi2.set_suffix_value(m.dof_v, 1)
    #m.pi3.set_suffix_value(m.dof_v, 1)
    m.rh_name = Suffix(direction=Suffix.IMPORT)
    
    ip = SolverFactory('ipopt_sens')
    ip.options["halt_on_ampl_error"] = "yes"
    ip.options["max_iter"] = 5000
    ip.options["tol"] = 1e-8
    ip.options["linear_Solver"] = "ma57"
    ip.options["compute_red_hessian"] = "yes"
    ip.options["nlp_scaling_method"] = "none"
    
    m.red_hessian = Suffix(direction = Suffix.EXPORT)
    m.pi1.set_suffix_value(m.red_hessian, 1)
    m.pi2.set_suffix_value(m.red_hessian, 2)
    #m.pi3.set_suffix_value(m.red_hessian, 3)
    ip.solve(m,tee=True)
    

    #m.pi3.pprint()
    
    k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
    k_aug.options["compute_inv"] = ""
    k_aug.options["no_barrier"] = ""
    k_aug.options["no_scale"] = ""
    m.ipopt_zL_in.update(m.ipopt_zL_out)
    m.ipopt_zU_in.update(m.ipopt_zU_out)   
    
    k_aug.solve(m,tee=True)
    _PI = {}
    PI_indices = {}
    for i in range(1,3):
        par = getattr(m,'pi'+str(i))  
        aux = par.get_suffix_value(m.rh_name)
        PI_indices[par.name] = aux
            
    with open("inv_.in", "r") as rh:
        ll = []
        l = rh.readlines()
        row = 0
        for i in l:
            ll = i.split()
            col = 0
            for j in ll:
                _PI[row, col] = float(j)
                col += 1
            row += 1
        rh.close()
    
    m.pi1.pprint()
    m.pi2.pprint()
    
    dimension = 2
    rows = {}
    confidence = chi2.isf(1-0.95,2) 
    A_dict = _PI
    center = [0,0]
    for m in range(dimension):
        rows[m] = np.array([A_dict[(m,i)] for i in range(dimension)])
    A = np.array([np.array(rows[i]) for i in range(dimension)])
    center = np.array([0]*dimension)
    U, s, V = np.linalg.svd(A) # singular value decomposition 
    radii = sqrt(confidence)*np.sqrt(s) # radii
    
    # transform in polar coordinates for simple plot
    theta = np.linspace(0.0, 2.0 * np.pi, 100) # 
    x = radii[0] * np.sin(theta) #
    y = radii[1] * np.cos(theta) #
    for i in range(len(x)):
        [x[i],y[i]] = np.dot([x[i],y[i]], V) + center
    plt.plot(x,y,label=str(N))
plt.legend()


# I DO NOT UNDERSTAND! #
# FUCK OFF KAUG IS WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ################
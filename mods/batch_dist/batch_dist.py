#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:12:05 2018

@author: flemmingholtorf
"""

from pyomo.core import *
from pyomo.environ import *
import numpy as np
from pyomo.dae import *
from pyomo.opt import ProblemFormat
from aux.cpoinsc import collptsgen
from aux.lagrange_f import lgr, lgry, lgrdot, lgrydot
import collections
import matplotlib.pyplot as plt
from six import itervalues, iterkeys, iteritems
import sys, csv



# Diehl Batch Destillation Column

class mod(ConcreteModel):
    def __init__(self, nfe, ncp, **kwargs):
        ConcreteModel.__init__(self)   
        self.nfe = nfe
        self.ncp = ncp
        self.N = 5
        self.delta_t = Param(initialize=1.28/nfe)
        self.tau_t = collptsgen(ncp, 1, 0) #compute normalized lagrange interpolation polynomial values according to desired collocation scheme
        
        # start at zero
        self.tau_i_t = {0: 0.}
        # create a list
        for ii in range(1, ncp + 1):
            self.tau_i_t[ii] = self.tau_t[ii - 1]
        
        
        
        # collocation
        self.fe_t = Set(initialize=[i for i in range(1,nfe+1)])
        self.cp = Set(initialize=[i for i in range(ncp+1)])
        
        self.ldot_t = Param(self.cp, self.cp, initialize=(lambda self, j, k: lgrdot(k, self.tau_i_t[j], ncp, 1, 0)))  #: watch out for this!
        self.l1_t = Param(self.cp, initialize=(lambda self, j: lgr(j, 1, ncp, 1, 0)))
        
        # Trays
        self.t = Set(initialize=[i for i in range(self.N+2)]) # stages 0 to N+1
            
        self.L = Var(self.fe_t, initialize=1.0)
        self.R = Var(self.fe_t, initialize=1.0, bounds=(0.0,15.0))
        self.M0 = Var(self.fe_t, self.cp, initialize=100.0, bounds=(0.0,1e3))
        self.dM0_dt = Var(self.fe_t, self.cp, initialize=-0.1)
        self.x = Var(self.fe_t, self.cp, self.t, initialize = 1.0, bounds=(0.0,1.0))
        self.dx_dt = Var(self.fe_t, self.cp, self.t, initialize = 0.0)
        self.y = Var(self.fe_t, self.cp, self.t, initialize = 0.5)
        self.MD = Var(self.fe_t, self.cp, initialize=1.0, bounds=(0.0,1e3))
        self.dMD_dt = Var(self.fe_t, self.cp, initialize=0.1)
        self.xD = Var(self.fe_t, self.cp, initialize=1.0, bounds=(0.0,1.0))
        self.dxD_dt = Var(self.fe_t, self.cp, initialize=0.01)
        
        # constant relative volatility alpha
        self.alpha = Param(initialize=6.0)
        self.p_alpha_par = Param(initialize=1.0, mutable = True)
        self.p_alpha = Var(initialize=1.0)
        # constant holdup of condenser
        self.Mc = Param(initialize = 0.1) 
        # constant MD
        self.V = Param(initialize=100.0) # 100 kmol/hr
        self.p_V_par = Param(initialize=1.0, mutable = True)
        self.p_V = Var(initialize=1.0)
        # constant holdup per stage
        self.m = Param(initialize=0.1)

        # initial values
        self.x_ic = Param(self.t, initialize = {i:1.0 for i in self.t},mutable=True)
        self.x_ic[0] = 0.5
        self.M0_ic = 100.0
        self.MD_ic = 0.1
        self.xD_ic = 1.0
        
        # slacks
        self.s_purity = Var(initialize=0.0, bounds = (0.0,None))
        
        # backoff
        self.xi_purity = Param(initialize=0.0, mutable=True)
###############################################################################
        # system dynamics

###############################################################################       
        #M0
        def _ode_M0(self,i,j):
            if j > 0:
                return self.dM0_dt[i,j] == self.L[i] - self.V*self.p_V
            else:
                return Constraint.Skip
        
        self.de_M0 = Constraint(self.fe_t, self.cp, rule=_ode_M0)
          
        def _collocation_M0(self,i,j):  
            if j > 0:
                return self.dM0_dt[i, j] == \
                       1/self.delta_t*sum(self.ldot_t[j, k] * self.M0[i, k] for k in self.cp)
            else:
                return Constraint.Skip
                
        self.dvar_t_M0 = Constraint(self.fe_t, self.cp, rule=_collocation_M0)
            
        def _continuity_M0(self, i):
            if i < nfe and nfe > 1:
                return self.M0[i + 1, 0] - sum(self.l1_t[j] * self.M0[i, j] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_M0 = Expression(self.fe_t, rule=_continuity_M0)
        self.cp_M0 = Constraint(self.fe_t, rule=lambda self, i: self.noisy_M0[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_M0(self):
            return self.M0[1, 0] - self.M0_ic

        self.M0_ice = Expression(rule = _init_M0)
        self.M0_icc = Constraint(rule =lambda self: 0.0 == self.M0_ice)
###############################################################################        
        #x
        def _ode_x(self,i,j,t):
            if j > 0:
                if t == 0:
                    return self.dx_dt[i,j,t] == 1/self.M0[i,j]*(self.L[i]*self.x[i,j,t+1]-self.V*self.p_V*self.y[i,j,t]+(self.V*self.p_V-self.L[i])*self.x[i,j,t])
                elif t == self.N+1:
                    return self.dx_dt[i,j,t] == 1/self.Mc*self.V*self.p_V*(self.y[i,j,t-1]-self.x[i,j,t])
                else:
                    return self.dx_dt[i,j,t] == 1/self.m*(self.L[i]*self.x[i,j,t+1]-self.V*self.p_V*self.y[i,j,t]+self.V*self.p_V*self.y[i,j,t-1]-self.L[i]*self.x[i,j,t])
            else:
                return Constraint.Skip
        
        self.de_x = Constraint(self.fe_t, self.cp, self.t, rule=_ode_x)
          
        def _collocation_x(self,i,j,t):  
            if j > 0:
                return self.dx_dt[i, j, t] == \
                       1/self.delta_t*sum(self.ldot_t[j, k] * self.x[i, k, t] for k in self.cp)
            else:
                return Constraint.Skip
                
        self.dvar_t_x = Constraint(self.fe_t, self.cp, self.t, rule=_collocation_x)
            
        def _continuity_x(self, i, t):
            if i < nfe and nfe > 1:
                return self.x[i + 1, 0, t] - sum(self.l1_t[j] * self.x[i, j, t] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_x = Expression(self.fe_t, self.t, rule=_continuity_x)
        self.cp_x = Constraint(self.fe_t, self.t, rule=lambda self, i, t: self.noisy_x[i,t] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_x(self,t):
            return self.x[1, 0, t] - self.x_ic[t]
        
        
        self.x_ice = Expression(self.t, rule = _init_x)
        self.x_icc = Constraint(self.t, rule = lambda self,t: 0.0 == self.x_ice[t])

###############################################################################
        #MD
        def _ode_MD(self,i,j):
            if j > 0:
                return self.dMD_dt[i,j] == -self.L[i] + self.V*self.p_V
            else:
                return Constraint.Skip
        
        self.de_MD = Constraint(self.fe_t, self.cp, rule=_ode_MD)
          
        def _collocation_MD(self,i,j):  
            if j > 0:
                return self.dMD_dt[i, j] == \
                       1/self.delta_t*sum(self.ldot_t[j, k] * self.MD[i, k] for k in self.cp)
            else:
                return Constraint.Skip
                
        self.dvar_t_MD = Constraint(self.fe_t, self.cp, rule=_collocation_MD)
            
        def _continuity_MD(self, i):
            if i < nfe and nfe > 1:
                return self.MD[i + 1, 0] - sum(self.l1_t[j] * self.MD[i, j] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_MD = Expression(self.fe_t, rule=_continuity_MD)
        self.cp_MD = Constraint(self.fe_t, rule=lambda self, i: self.noisy_MD[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_MD(self):
            return self.MD[1, 0] - self.MD_ic
        
        self.MD_ice = Expression(rule = _init_MD)
        self.MD_icc = Constraint(rule =lambda self: 0.0 == self.MD_ice)
###############################################################################
        #xD
        def _ode_xD(self,i,j):
            if j > 0:
                return self.dxD_dt[i,j] == (self.V*self.p_V - self.L[i])/self.MD[i,j]*(self.x[i,j,self.N+1]-self.xD[i,j])
            else:
                return Constraint.Skip
        
        self.de_xD = Constraint(self.fe_t, self.cp, rule=_ode_xD)
          
        def _collocation_xD(self,i,j):  
            if j > 0:
                return self.dxD_dt[i, j] == \
                       1/self.delta_t*sum(self.ldot_t[j, k] * self.xD[i, k] for k in self.cp)
            else:
                return Constraint.Skip
                
        self.dvar_t_xD = Constraint(self.fe_t, self.cp, rule=_collocation_xD)
            
        def _continuity_xD(self, i):
            if i < nfe and nfe > 1:
                return self.xD[i + 1, 0] - sum(self.l1_t[j] * self.xD[i, j] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_xD = Expression(self.fe_t, rule=_continuity_xD)
        self.cp_xD = Constraint(self.fe_t, rule=lambda self, i: self.noisy_xD[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_xD(self):
            return self.xD[1,0] - self.xD_ic
        
        self.xD_ice = Expression(rule = _init_xD)
        self.xD_icc = Constraint(rule = lambda self: 0.0 == self.xD_ice)

###############################################################################
        # algebraic eqns
        def _phase_equilibrium(self,i,j,t):
            if j > 0:
                return 0.0 == self.x[i,j,t]*self.alpha*self.p_alpha - self.y[i,j,t]*(self.x[i,j,t]*(self.alpha*self.p_alpha-1.0)+1.0)
            else:
                return Constraint.Skip
            
        self.phase_equlibrium = Constraint(self.fe_t, self.cp, self.t, rule = _phase_equilibrium)
        
        def _reflux(self,i):
            return 0.0 == self.R[i]*self.V*self.p_V - (1.0 + self.R[i])*self.L[i]
        
        self.reflux = Constraint(self.fe_t, rule=_reflux)
###############################################################################
        # constraints
        def _epc_purity(self):
            return 0.99 - self.xD[self.nfe,self.ncp] + self.s_purity + self.xi_purity == 0.0
        
        self.epc_purity = Constraint(rule=_epc_purity)
        
        # dummy constraint
        def _dummy_constraint_p_alpha(self):
            return self.p_alpha == self.p_alpha_par
        
        self.dummy_constraint_p_alpha = Constraint(rule=_dummy_constraint_p_alpha)
        
        def _dummy_constraint_p_V(self):
            return self.p_V == self.p_V_par
        
        self.dummy_constraint_p_V = Constraint(rule=_dummy_constraint_p_V)
###############################################################################
        # objective        
        def _eobj(self):
            return -self.MD[self.nfe,self.ncp]
        
        self.eobj = Objective(rule=_eobj)

###############################################################################
        # Suffixes
        self.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)  
        
    def write_nl(self):
        """Writes the nl file and the respective row & col"""
        name = str(self.__class__.__name__) + ".nl"
        self.write(filename=name,
                   format=ProblemFormat.nl,
                   io_options={"symbolic_solver_labels": True})
        
#ip = SolverFactory('ipopt')
#ip.options["linear_solver"] = "ma57"
#m = mod(30,3)
#ip.solve(m, tee = True)
#
#R_aux = [(m.R[i].value,m.R[i].value) for i in m.fe_t]
#R = []
#t = []
#for i in range(m.nfe):
#    t.append(i*m.delta_t.value)
#    t.append(t[2*i]+m.delta_t.value)
#    R.append(R_aux[i][0])
#    R.append(R_aux[i][1])
#plt.figure(1)
#plt.plot(t,R)
#
#xD = [m.xD[i].value for i in m.xD.index_set()]
#t_coll = [m.delta_t*(i+m.tau_i_t[cp]) for i in m.fe_t for cp in m.cp]
#
#plt.figure(2)
#plt.plot(t_coll,xD)
#
## Robust approx.
#iters = 0
#iterlim = 10
#converged = False
#eps = 1e-5
#alpha = 0.278
#p_noisy = {'alpha':[()]}
#cons = ['purity']
#
#k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
#k_aug.options["compute_dsdp"] = ""
#
#n_p = 0
#for key1 in p_noisy:
#    for key2 in p_noisy[key1]:
#        n_p += 1
#
#backoff = {}
#for i in cons:
#    backoff_var = getattr(m,'xi_'+i)
#    for index in backoff_var.index_set():
#        try:
#            backoff[('s_'+i,index)] = 0.0
#            backoff_var[index].value = 0.0
#        except KeyError:
#            continue
#
#while (iters < iterlim and not(converged)):
#    # solve optimization problem
#    for i in cons:
#        slack = getattr(m, 's_'+i)
#        for index in slack.index_set():
#            slack[index].setlb(0)
#    m.R.unfix()
#    ip.solve(m, tee=True)
#    m.ipopt_zL_in.update(m.ipopt_zL_out)
#    m.ipopt_zU_in.update(m.ipopt_zU_out)
#    if iters == 0:
#        m.var_order = Suffix(direction=Suffix.EXPORT)
#        m.dcdp = Suffix(direction=Suffix.EXPORT)
#        i = 1
#        reverse_dict_pars = {}
#        for p in p_noisy:
#            for key in p_noisy[p]:
#                if key != ():
#                    dummy = 'dummy_constraint_p_' + p + '_' + key
#                else:
#                    dummy = 'dummy_constraint_p_' + p
#                dummy_con = getattr(m, dummy)
#                for index in dummy_con.index_set():
#                    m.dcdp.set_value(dummy_con[index], i)
#                    reverse_dict_pars[i] = (p,key)
#                    i += 1
#    
#        i = 1
#        reverse_dict_cons = {}
#        for k in cons:
#            s = getattr(m, 's_'+k)
#            for index in s.index_set():
#                if not(s[index].stale):
#                    m.var_order.set_value(s[index], i)
#                    reverse_dict_cons[i] = ('s_'+ k,index)
#                    i += 1
#    
#    # get sensitivities
#    m.R.fix()   
#    m.s_purity.setlb(None)
#    k_aug.solve(m, tee=True)
#    sys.exit()
#
#    sens = {}
#    with open('dxdp_.dat') as f:
#        reader = csv.reader(f, delimiter="\t")
#        i = 1
#        for row in reader:
#            k = 1
#            s = reverse_dict_cons[i]
#            for col in row[1:]:
#                p = reverse_dict_pars[k]
#                sens[(s,p)] = float(col)
#                k += 1
#            i += 1
#            
#    
#    # convergence check and update    
#    converged = True
#    for i in cons:
#        backoff_var = getattr(m,'xi_'+i)
#        for index in backoff_var.index_set():
#            try:
#                new_backoff = sum(abs(alpha*sens[(('s_'+i,index),reverse_dict_pars[k])]) for k in range(1,n_p+1))
#                old_backoff = backoff[('s_'+i,index)]
#                if backoff[('s_'+i,index)] - new_backoff < 0:
#                    backoff[('s_'+i,index)] = new_backoff
#                    backoff_var[index].value = new_backoff
#                    if old_backoff - new_backoff < -eps:
#                        converged = False
#                else:
#                    continue
#            except KeyError:
#                continue
#    iters += 1
#
#for i in cons:
#    slack = getattr(m, 's_'+i)
#    for index in slack.index_set():
#        slack[index].setlb(0)
#m.R.unfix()
#ip.solve(m, tee=True)
#
#R_aux = [(m.R[i].value,m.R[i].value) for i in m.fe_t]
#R = []
#t = []
#for i in range(m.nfe):
#    t.append(i*m.delta_t.value)
#    t.append(t[2*i]+m.delta_t.value)
#    R.append(R_aux[i][0])
#    R.append(R_aux[i][1])
#plt.figure(1)
#plt.plot(t,R,label='approx')
#plt.legend()
#
#xD = [m.xD[i].value for i in m.xD.index_set()]
#t_coll = [m.delta_t*(i+m.tau_i_t[cp]) for i in m.fe_t for cp in m.cp]
#
#plt.figure(2)
#plt.plot(t_coll,xD,label='approx')
#plt.legend()
#
## solve real problem
#m.xi_purity = 0.0
#m.p_alpha_par = 1.0 + alpha
#m.p_V_par = 1.0
#m.R.fix()
#m.epc_purity.deactivate()
#ip.solve(m,tee=False)
#R_aux = [(m.R[i].value,m.R[i].value) for i in m.fe_t]
#R = []
#t = []
#for i in range(m.nfe):
#    t.append(i*m.delta_t.value)
#    t.append(t[2*i]+m.delta_t.value)
#    R.append(R_aux[i][0])
#    R.append(R_aux[i][1])
#plt.figure(1)
#plt.plot(t,R,label='ub')
#plt.legend()
#
#xD = [m.xD[i].value for i in m.xD.index_set()]
#t_coll = [m.delta_t*(i+m.tau_i_t[cp]) for i in m.fe_t for cp in m.cp]
#
#plt.figure(2)
#plt.plot(t_coll,xD,label='ub')
#plt.legend()
#
#m.p_alpha_par = 1.0 - alpha
#m.p_V_par = 1.0
#ip.solve(m,tee=False)
#R_aux = [(m.R[i].value,m.R[i].value) for i in m.fe_t]
#R = []
#t = []
#for i in range(m.nfe):
#    t.append(i*m.delta_t.value)
#    t.append(t[2*i]+m.delta_t.value)
#    R.append(R_aux[i][0])
#    R.append(R_aux[i][1])
#plt.figure(1)
#plt.plot(t,R,label='lb')
#plt.legend()
#
#xD = [m.xD[i].value for i in m.xD.index_set()]
#t_coll = [m.delta_t*(i+m.tau_i_t[cp]) for i in m.fe_t for cp in m.cp]
#
#plt.figure(2)
#plt.plot(t_coll,xD,label='lb')
#plt.legend()
#
#m.epc_purity.activate()
#m.R.unfix()
#ip.solve(m,tee=True)
#
#R_aux = [(m.R[i].value,m.R[i].value) for i in m.fe_t]
#R = []
#t = []
#for i in range(m.nfe):
#    t.append(i*m.delta_t.value)
#    t.append(t[2*i]+m.delta_t.value)
#    R.append(R_aux[i][0])
#    R.append(R_aux[i][1])
#plt.figure(1)
#plt.plot(t,R,label='nonlin')
#plt.legend()
#
#xD = [m.xD[i].value for i in m.xD.index_set()]
#t_coll = [m.delta_t*(i+m.tau_i_t[cp]) for i in m.fe_t for cp in m.cp]
#
#plt.figure(2)
#plt.plot(t_coll,xD,label='nonlin')
#plt.legend()
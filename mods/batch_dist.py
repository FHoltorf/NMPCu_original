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
import sys



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
        self.R = Var(self.fe_t, initialize=1.0)
        self.M0 = Var(self.fe_t, self.cp, initialize=1.0)
        self.dM0_dt = Var(self.fe_t, self.cp, initialize=1.0)
        self.y0 = Var(self.fe_t, self.cp, initialize=1.0)
        self.x = Var(self.fe_t, self.cp, self.t, initialize = 1.0)
        self.dx_dt = Var(self.fe_t, self.cp, self.t, initialize = 1.0)
        self.y = Var(self.fe_t, self.cp, self.t, initialize = 0.0)
        self.MD = Var(self.fe_t, self.cp, initialize=1.0)
        self.dMD_dt = Var(self.fe_t, self.cp, initialize=1.0)
        self.xD = Var(self.fe_t, self.cp, initialize=1.0)
        self.dxD_dt = Var(self.fe_t, self.cp, initialize=1.0)
        
        # constant relative volatility alpha
        self.alpha = Param(initialize=6.0)
        # constant holdup of condenser
        self.Mc = Param(initialize = 0.1) 
        # constant MD
        self.V = Param(initialize=100.0) # 100 kmol/hr
        # constant holdup per stage
        self.m = Param(initialize=0.1)

        # initial values
        self.x_ic = Param(self.t, initialize = {i:1.0 for i in self.t},mutable=True)
        self.x_ic[0] = 0.5
        self.M0_ic = 100.0
        self.MD_ic = 0.1
        self.xD_ic = 0.01
        
###############################################################################
        # system dynamics

###############################################################################       
        #M0
        def _ode_M0(self,i,j):
            if j > 0:
                return self.dM0_dt[i,j] == self.L[i] - self.V
            else:
                return Constraint.Skip
        
        self.de_M0 = Constraint(self.fe_t, self.cp, rule=_ode_M0)
          
        def _collocation_M0(self,i,j):  
            if j > 0:
                return self.dM0_dt[i, j] == \
                       self.delta_t*sum(self.ldot_t[j, k] * self.M0[i, k] for k in self.cp)
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
                    return self.dx_dt[i,j,t] == 1/self.M0[i,j]*(self.L[i]*self.x[i,j,t+1]-self.V*self.y[i,j,t]+(self.V-self.L[i])*self.x[i,j,t])
                elif t == self.N+1:
                    return self.dx_dt[i,j,t] == 1/self.Mc*self.V*(self.y[i,j,t-1]-self.x[i,j,t])
                else:
                    return self.dx_dt[i,j,t] == 1/self.m*(self.L[i]*self.x[i,j,t+1]-self.V*self.y[i,j,t]+self.V*self.y[i,j,t-1]-self.L[i]*self.x[i,j,t])
            else:
                return Constraint.Skip
        
        self.de_x = Constraint(self.fe_t, self.cp, self.t, rule=_ode_x)
          
        def _collocation_x(self,i,j,t):  
            if j > 0:
                return self.dx_dt[i, j, t] == \
                       self.delta_t*sum(self.ldot_t[j, k] * self.x[i, k, t] for k in self.cp)
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
                return self.dMD_dt[i,j] == -self.L[i] + self.V
            else:
                return Constraint.Skip
        
        self.de_MD = Constraint(self.fe_t, self.cp, rule=_ode_MD)
          
        def _collocation_MD(self,i,j):  
            if j > 0:
                return self.dMD_dt[i, j] == \
                       self.delta_t*sum(self.ldot_t[j, k] * self.MD[i, k] for k in self.cp)
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
                return self.dxD_dt[i,j] == (self.V - self.L[i])/self.MD[i,j]*(self.x[i,j,self.N+1]-self.xD[i,j])
            else:
                return Constraint.Skip
        
        self.de_xD = Constraint(self.fe_t, self.cp, rule=_ode_xD)
          
        def _collocation_xD(self,i,j):  
            if j > 0:
                return self.dxD_dt[i, j] == \
                       self.delta_t*sum(self.ldot_t[j, k] * self.xD[i, k] for k in self.cp)
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
        self.xD_icc = Constraint(rule =lambda self: 0.0 == self.xD_ice)

###############################################################################
        # algebraic eqns
        def _phase_equilibrium(self,i,j,t):
            if j > 0:
                return 0.0 == self.x[i,j,t]*self.alpha - self.y[i,j,t]*(self.x[i,j,t]*(self.alpha-1.0)+1.0)
            else:
                return Constraint.Skip
            
        self.phase_equlibrium = Constraint(self.fe_t, self.cp, self.t, rule = _phase_equilibrium)
        
        def _reflux(self,i):
            return 0.0 == self.R[i]*self.V - (1.0 + self.R[i])*self.L[i]
        
###############################################################################
        # constraints
        def _epc_purity(self):
            return 0.99 - self.xD[self.nfe,self.ncp] <= 0.0
        

m = mod(20,3)

            
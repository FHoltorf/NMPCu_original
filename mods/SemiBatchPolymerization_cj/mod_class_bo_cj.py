# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:37:31 2017
@author: FlemmingHoltorf
"""
#This file includes the moment model of propoxylation processes for recipe optimization. 
#This is also to illustrate the element-by-element initialization scheme.

# Work in progress NOTES:
# -24 elements
# -3 moments
from pyomo.core import *
from pyomo.environ import *
from six import itervalues, iterkeys, iteritems
from pyomo.dae import *
from pyomo.opt import ProblemFormat
from aux.cpoinsc import collptsgen
from aux.lagrange_f import lgr, lgry, lgrdot, lgrydot
import collections, sys
import matplotlib.pyplot as plt
import numpy as np

class SemiBatchPolymerization(ConcreteModel):
    def __init__(self, nfe, ncp, **kwargs):        
        ConcreteModel.__init__(self)        
        # scaling factors
        self.W_scale = 1.0
        self.Y_scale = 1.0e-2
        self.PO_scale = 1.0e1
        self.MY0_scale = 1.0e-1
        self.MX0_scale = 1.0e1
        self.MX1_scale = 1.0e2
        self.MW_scale = 1.0e2
        self.X_scale = 1.0
        self.m_tot_scale = 1.0e4
        self.T_scale = 1.0e2
        self.Tad_scale = 1.0e2
        self.Vi_scale = 1.0e-2
        self.PO_fed_scale = 1.0e2
        self.int_T_scale = 1.0e2
        self.int_Tad_scale = 1.0e2
        self.G_scale = 1.0
        self.U_scale = 1.0e-2
        self.monomer_cooling_scale = 1.0e-2
        self.scale = 1.0
    
        # collocation pts
        self.nfe = nfe
        self.ncp = ncp
        self.tau_t = collptsgen(ncp, 1, 0) #compute normalized lagrange interpolation polynomial values according to desired collocation scheme
        
        # start at zero
        self.tau_i_t = {0: 0.}
        # create a list
        for ii in range(1, ncp + 1):
            self.tau_i_t[ii] = self.tau_t[ii - 1]
               
        # specify
        self.tf = Var(initialize=9.6*60/nfe,bounds=(2*60,9.20*60)) # batch time in [min]
        
        # sets:
        self.o = Set(initialize=[i for i in range(2)]) # moments
        self.r = Set(initialize=['a','i','p','t']) # reactions
        self.fe_t = Set(initialize=[i for i in range(1,nfe+1)])
        self.cp = Set(initialize=[i for i in range(ncp+1)])
        self.epc = Set(initialize=[i for i in range(1,5)])
        self.pc = Set(initialize=[i for i in range(1,4)])
        
        # parameters for l1-relaxation of endpoint-constraints
        self.eps = Var(self.epc, initialize=0.0, bounds=(0,None))
        #self.eps.fix()
        self.eps_pc = Var(self.fe_t, self.cp, self.pc, initialize=0.0, bounds=(0,None))
        self.rho = Param(initialize=1e6, mutable=True)
        self.gamma = Param(initialize=10.0,mutable=True)
        
        # auxilliary parameter to enable non-uniform finite element distribution
        self.fe_dist = Param(self.fe_t, initialize = 1.0, mutable=True)
        
        # lagrange polynomials and time derivative at radau nodes
        self.ldot_t = Param(self.cp, self.cp, initialize=(lambda self, j, k: lgrdot(k, self.tau_i_t[j], ncp, 1, 0)))  #: watch out for this!
        self.l1_t = Param(self.cp, initialize=(lambda self, j: lgr(j, 1, ncp, 1, 0)))
        
        # physiochemical properties
        self.mw_H2O = Param(initialize=18.02) # [g/mol] or [kg/kmol] molecular weight of H2O 
        self.mw_PO = Param(initialize=58.08) # [g/mol] or [kg/kmol] molecular weight of PO
        self.mw_KOH = Param(initialize=56.11) # [g/mol] or [kg/kmol] molecular weight of KOH
        self.mw_PG = Param(initialize=76.09) # [g/mol] or [kg/kmol] molecular weight of PG
        self.num_OH = Param(initialize=2) # [-] number of OH groups
        
        # constants
        self.Rg = Param(initialize=8.314472e-3) # [kJ/mol/K] universal gas constant
        self.Tb = Param(initialize=273.15) # [K] Celsius 0;
        
        # thermodynamic properties (selfodel parameter)
        self.bulk_cp_1 = Param(initialize=1.1) # [kJ/kg/K]  
        self.bulk_cp_2 = Param(initialize=2.72e-3) # [kJ/kg/K^2]
        self.mono_cp_1 = Param(initialize=0.92) #53.347) # [kJ/kg/K]
        self.mono_cp_2 = Param(initialize=8.87e-3)#5.1543e-1) # [kJ/kmol/K^2]
        self.mono_cp_3 = Param(initialize=-3.10e-5)#-1.8029e-3) # [kJ/kmol/K^3]
        self.mono_cp_4 = Param(initialize=4.78e-8)#2.7795e-6 # [kJ/kmol/K^4]
        
        # batch charge conditions
        self.m_H2O = Param(initialize=180.98) # [kg] mass of H2O
        self.m_PO = Param(initialize=30452.76) # [kg] mass of PO
        self.m_KOH = Param(initialize=151.50) # [kg] mass of KOH
        self.m_PG = Param(initialize=1051.88) # [kg] mass of PG
        self.m_total = Param(initialize=self.m_H2O+self.m_PO+self.m_KOH+self.m_PG) # [kg] total mass in the reactor
        self.n_H2O = Param(initialize=self.m_H2O/self.mw_H2O) # [kmol] mole of H2O
        self.n_PO = Param(initialize=self.m_PO/self.mw_PO) # [kmol] mole of PO
        self.n_KOH = Var(initialize=self.m_KOH/self.mw_KOH) # [kmol] mole of KOH
        self.n_KOH.fix() # this does not make sense to be time-variant therefore no p_n_KOH necessary
        self.n_PG = Param(initialize=self.m_PG/self.mw_PG) # [kmol] mole of PG;
        
        # reactor and product specs
        self.T_safety = Param(initialize=170.0) #190 [°C] maximum allowed temperature after adiabatic temperature rise
        self.T_max = Param(initialize=150.0)
        self.T_min = Param(initialize=100.0)
        self.molecular_weight = Param(initialize=949.5, mutable=True) # 3027.74 # [g/mol] or [kg/kmol] target molecular weights
        self.molecular_weight_max = Param(initialize=949.5+20, mutable=True)
        self.unsat_value = Param(initialize=0.032)#-0.0018) #0.032 # unsaturation value
        self.unreacted_PO = Param(initialize=120.0) #120.0 # [PPM] unreacted PO
        self.rxr_volume = Param(initialize=41.57) # [m^3] volume of the reactor
        self.rxr_pressure = Param(initialize=253) # [kPa] initial pressure
        self.rxr_temperature = Param(initialize=122.9) # [°C] initial temperature 
        self.feed_temp = Param(initialize=25+self.Tb) # [K] feed temperature of monomers
        
        # polymerization kinetics
        aux = np.array([8.64e4,3.964e5,1.35042e4,1.509e6]) # [m^3/mol/s]
        self.A = Var(self.r,initialize=({'a':aux[0]/self.scale,'i':aux[1]/self.scale,'p':aux[2]/self.scale,'t':aux[3]/self.scale}), bounds = (1e4,None)) #
        self.A.fix()
        self.p_A = Var(self.fe_t, self.r, initialize=1.0) # this makes sense to be time-variant
        self.p_A.fix()
        self.Ea = Param(self.r,initialize=({'a':82.425,'i':77.822,'p':69.172,'t':105.018}), mutable=True) # [kJ/mol] activation engergy 
        self.Hrxn = Param(self.r, initialize=({'a':0, 'i':92048, 'p':92048,'t':0}), mutable=True)
        self.Hrxn_aux = Var(self.r, initialize=({'a':1.0, 'i':1.0, 'p':1.0,'t':1.0}))
        self.Hrxn_aux.fix()
        self.p_Hrxn_aux = Var(self.fe_t, self.r, initialize=1.0) # this makes sense to be time-variant
        self.p_Hrxn_aux.fix()
        self.max_heat_removal = Param(initialize=2.2e3/self.Hrxn['p']*60, mutable=True) # [kmol (PO)/min] maximum amount of heat removal rate scaled by Hrxn('p') (see below)s
        
        # heat transfer
        self.kA = Var(initialize=2200.0/self.Hrxn['p']*60.0/20.0) # [kW/K]
        self.kA.fix()
        self.p_kA = Var(self.fe_t, initialize=1.0) # this makes sense to be time-variant
        self.p_kA.fix()

        # parameters for initializing differential variabales
        self.W_ic = Param(initialize=self.n_H2O/self.W_scale, mutable=True)
        self.PO_ic = Param(initialize=0.0, mutable=True)
        self.m_tot_ic = Param(initialize=(self.m_PG+self.m_KOH+self.m_H2O)/self.m_tot_scale, mutable=True)
        self.X_ic = Param(initialize=(self.n_PG*self.num_OH+self.n_H2O*self.num_OH)/self.X_scale, mutable=True)
        self.Y_ic = Param(initialize=0.0, mutable=True)
        self.MY_ic = Param(initialize=0.0, mutable=True)
        self.MX_ic = Param(self.o, initialize={0:0.0,1:0.0}, mutable=True)
        self.PO_fed_ic = Param(initialize=0.0, mutable=True) 
        self.T_ic = Param(initialize=393.15/self.T_scale, mutable=True) 
        self.T_cw_ic = Param(initialize=373.15/self.T_scale, mutable=True) 

        # variables
        # decision variables/controls (piece wise constant)
        #piecewise constant controls
        self.F = Var(self.fe_t, initialize=1,bounds=(0.0,None))
        self.dT_cw_dt = Var(self.fe_t, initialize=0.0)
        self.u1 = Var(self.fe_t, initialize=1.0,bounds=(0.0,None))
        self.u2 = Var(self.fe_t, initialize=1.0,bounds=(0.0,None))
        
        # differential variables
        # cooling water
        self.T_cw = Var(self.fe_t, self.cp, initialize=self.T_cw_ic, bounds=(0.0,None))
        
        # reactor temperature
        self.T = Var(self.fe_t, self.cp, initialize=385.0/self.T_scale) # temperature
        self.dT_dt = Var(self.fe_t, self.cp, initialize=0.0)
        
        # water
        self.W = Var(self.fe_t, self.cp, initialize=self.W_ic, bounds=(0.0,None)) # Water
        self.dW_dt = Var(self.fe_t, self.cp)

        # monomer
        self.PO = Var(self.fe_t, self.cp, initialize=self.PO_ic, bounds=(0.0,None)) # propylene oxide
        self.dPO_dt = Var(self.fe_t,self.cp)

        # total mass  --> removed ODE and replaced by one-to-one relation with PO_fed (since better scaled that way)
        self.m_tot = Var(self.fe_t, self.cp, initialize=self.m_tot_ic, bounds=(0,None)) # total mass
        #self.dm_tot_dt = Var(self.fe_t, self.cp)

        # pseudo species (product) --> removed ODE and replaced by reaction invariant eqn
        self.X = Var(self.fe_t, self.cp, initialize=self.X_ic, bounds=(0.0,None)) # saturated product
        #self.dX_dt = Var(self.fe_t, self.cp)
        
        # moment of product
        self.MX = Var(self.fe_t,self.cp,self.o, bounds=(0.0,None)) # moments of X
        self.dMX_dt = Var(self.fe_t,self.cp,self.o)
        
        # pseudospecies (byproduct)
        self.Y = Var(self.fe_t,self.cp, initialize=self.Y_ic, bounds=(0.0,None)) # unsaturated by-product
        self.dY_dt = Var(self.fe_t,self.cp)
      
        # moment of byproduct
        self.MY = Var(self.fe_t,self.cp, initialize=self.MY_ic, bounds=(0.0,None)) # moments of Y
        self.dMY_dt = Var(self.fe_t,self.cp)
        
        # fed monomer
        self.PO_fed = Var(self.fe_t,self.cp,initialize=self.PO_fed_ic, bounds=(0.0,None)) # total amount of PO fed
        self.dPO_fed_dt = Var(self.fe_t,self.cp)
        
        # reactions
        self.k_l = Var(self.fe_t, self.cp, self.r, initialize=0.0, bounds=(None,20)) # rate coefficients 
        self.kr = Var(self.fe_t,self.cp,self.r) # rate coefficients 
        
        # algebraic variables
        self.Vi = Var(self.fe_t,self.cp, bounds=(0,1e10)) # current volume 
        self.G = Var(self.fe_t,self.cp,bounds=(0.0,None)) # no idea
        self.U = Var(self.fe_t,self.cp,bounds=(0.0,None)) # heat transfer coefficient 
        self.MG = Var(self.fe_t,self.cp) # no idea
        
        # thermodynamic
        self.int_T = Var(self.fe_t, self.cp, initialize=653/self.int_T_scale) # internal energy
        self.int_Tad = Var(self.fe_t, self.cp, initialize=700/self.int_Tad_scale) # internal energy after adiabatic depletion of monomer
        self.Tad = Var(self.fe_t, self.cp, initialize=417.7/self.Tad_scale) 
        self.f = Var(self.fe_t, self.cp, initialize=1.121)
        self.monomer_cooling = Var(self.fe_t, self.cp, initialize=0.08) 
        self.heat_removal = Var(self.fe_t, self.cp, initialize=0.0)
        self.Qr = Var(self.fe_t,self.cp, initialize=0.0)
        self.Qc = Var(self.fe_t,self.cp, initialize=0.0)
        
        # define slack variables
        # path constraints
        self.s_temp_b = Var(self.fe_t, self.cp, initialize = 0, bounds=(0,None))
        self.s_T_max = Var(self.fe_t, self.cp, initialize=0.0, bounds=(0,None))
        self.s_T_min = Var(self.fe_t, self.cp, initialize=0.0, bounds=(0,None))
        # endpoint constraints
        self.s_mw = Var(initialize=0, bounds=(0,None))
        self.s_PO_ptg = Var(initialize=0, bounds=(0,None))
        self.s_unsat = Var(initialize=0, bounds=(0,None))
        self.s_PO_fed = Var(initialize=0, bounds=(0,None))
        self.s_mw_ub = Var(initialize=0, bounds=(0,None))

        # back-offs 
        #standard
        self.xi_mw_ub = Param(initialize=0.0, mutable=True)
        self.xi_mw = Param(initialize=3.0, mutable=True)#Param(initialize=0.18, mutable=True)#4.22 # 2.0 used for standard results
        self.xi_PO_ptg = Param(initialize=100.0, mutable=True)#Param(initialize=80.0, mutable=True)#26.51 #100.0 used for standard results
        self.xi_unsat = Param(initialize=0.008, mutable=True)# 0.00767363086751089 Param(initialize=0.0024, mutable=True)#0.0036 used for standard results
        self.xi_temp_b = Param(self.fe_t, self.cp, initialize=14.0, mutable=True) #Param(self.fe_t, self.cp, initialize=9.09, mutable=True)      #12.09  
        self.xi_T_max = Param(self.fe_t, self.cp, initialize = 0.4143011526048568, mutable=True) #Param(self.fe_t, self.cp, initialize=0.6, mutable=True)#1.8
        self.xi_T_min = Param(self.fe_t, self.cp, initialize=2.1042304467330464, mutable=True) #Param(self.fe_t, self.cp, initialize=1.2, mutable=True)#2.43   
        
        self.epc_indices = {1:'PO_ptg',2:'unsat',3:'mw',4:'mw_ub'}
        self.pc_indices = {1:'temp_b',2:'T_max',3:'T_min'}
        
        # closures
        def _total_mass_balance(self,i,j):
            if j > 0:
                return self.m_tot[i,j]*self.m_tot_scale == self.PO_fed[i,j]*self.PO_fed_scale*self.mw_PO+self.m_tot_ic*self.m_tot_scale
            else:
                return Constraint.Skip
        
        self.total_mass_balance = Constraint(self.fe_t, self.cp, rule=_total_mass_balance)
        
        def _rxn_invariant(self,i,j):
            if j>0:
                return 10.0432852386*2 + 47.7348816877 == self.W[i,j]*2*self.W_scale + self.X[i,j]*self.X_scale + self.MX[i,j,0]*self.MX0_scale
            else:
                return Constraint.Skip
        
        self.rxn_invariant = Constraint(self.fe_t,self.cp, rule=_rxn_invariant)
        
        # system dynamics
        # Water
        def _ode_W(self,i,j):
            if j > 0:
                return self.dW_dt[i,j] == -self.kr[i,j,'a']*self.W[i,j]*self.PO[i,j]*self.PO_scale*self.Vi[i,j]*self.Vi_scale # * self.W_scale/self.W_scale
            else:
                return Constraint.Skip
        self.de_W = Constraint(self.fe_t, self.cp, rule=_ode_W)
        
        
        def _collocation_W(self, i, j):
            if j > 0:
                return self.dW_dt[i, j] == \
                       sum(self.ldot_t[j, k] * self.W[i, k] for k in self.cp)
            else:
                return Constraint.Skip
        
        self.dvar_t_W = Constraint(self.fe_t, self.cp, rule=_collocation_W)
        
        def _continuity_W(self, i):
            if i < nfe and nfe > 1:
                return self.W[i + 1, 0] - sum(self.l1_t[j] * self.W[i, j] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_W = Expression(self.fe_t, rule=_continuity_W)
        self.cp_W = Constraint(self.fe_t, rule=lambda self,i:self.noisy_W[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_W(self):
            return self.W[1, 0] - self.W_ic
        
        self.W_ice = Expression(rule=_init_W)
        self.W_icc = Constraint(rule=lambda self: self.W_ice == 0.0)
        
        def _ode_PO(self,i,j):
            if j > 0:
                return self.dPO_dt[i,j] == (self.F[i]*self.tf*self.fe_dist[i] - (((self.kr[i,j,'i']-self.kr[i,j,'p'])*(self.G[i,j]*self.G_scale + self.U[i,j]*self.U_scale) + (self.kr[i,j,'p'] + self.kr[i,j,'t'])*self.n_KOH + self.kr[i,j,'a']*self.W[i,j]*self.W_scale)*self.PO[i,j]*self.PO_scale*self.Vi[i,j]*self.Vi_scale))/self.PO_scale
            else:
                return Constraint.Skip
        
        self.de_PO = Constraint(self.fe_t, self.cp, rule=_ode_PO)
          
        def _collocation_PO(self,i,j):  
            if j > 0:
                return self.dPO_dt[i, j] == \
                       sum(self.ldot_t[j, k] * self.PO[i, k] for k in self.cp)
            else:
                return Constraint.Skip
                
        self.dvar_t_PO = Constraint(self.fe_t, self.cp, rule=_collocation_PO)
            
        def _continuity_PO(self, i):
            if i < nfe and nfe > 1:
                return self.PO[i + 1, 0] - sum(self.l1_t[j] * self.PO[i, j] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_PO = Expression(self.fe_t, rule=_continuity_PO)
        self.cp_PO = Constraint(self.fe_t, rule=lambda self, i: self.noisy_PO[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_PO(self):
            return self.PO[1, 0] - self.PO_ic
        
        self.PO_ice = Expression(rule=_init_PO)
        self.PO_icc = Constraint(rule=lambda self: self.PO_ice == 0.0)
        
        def _ode_PO_fed(self,i,j):
            if j > 0:
                return self.dPO_fed_dt[i,j] == self.F[i]*self.tf*self.fe_dist[i]/self.PO_fed_scale
            else:
                return Constraint.Skip
        
        self.de_PO_fed = Constraint(self.fe_t, self.cp, rule=_ode_PO_fed)
          
        def _collocation_PO_fed(self,i,j):  
            if j > 0:
                return self.dPO_fed_dt[i, j] == \
                       sum(self.ldot_t[j, k] * self.PO_fed[i, k] for k in self.cp)
            else:
                return Constraint.Skip
                
        self.dvar_t_PO_fed = Constraint(self.fe_t, self.cp, rule=_collocation_PO_fed)
            
        def _continuity_PO_fed(self, i):
            if i < nfe and nfe > 1:
                return self.PO_fed[i + 1, 0] - sum(self.l1_t[j] * self.PO_fed[i, j] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_PO_fed = Expression(self.fe_t, rule=_continuity_PO_fed)
        self.cp_PO_fed = Constraint(self.fe_t, rule=lambda self, i: self.noisy_PO_fed[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_PO_fed(self):
            return self.PO_fed[1, 0] - self.PO_fed_ic

        self.PO_fed_ice = Expression(rule=_init_PO_fed)        
        self.PO_fed_icc = Constraint(rule=lambda self: self.PO_fed_ice == 0.0)

        
        def _ode_MX(self,i,j,o):
            if j > 0:   
                if o == 0:
                    return self.dMX_dt[i,j,o] == (self.kr[i,j,'i']*self.G[i,j]*self.G_scale*self.PO[i,j]*self.PO_scale*self.Vi[i,j]*self.Vi_scale)/self.MX0_scale
                elif o == 1:
                    return self.dMX_dt[i,j,o] == (self.kr[i,j,'i']*self.G[i,j]*self.G_scale+self.kr[i,j,'p']*self.MG[i,j])*self.PO[i,j]*self.PO_scale*self.Vi[i,j]*self.Vi_scale/self.MX1_scale
            else:
                return Constraint.Skip
        self.de_MX = Constraint(self.fe_t, self.cp, self.o, rule=_ode_MX)
        
        def _collocation_MX(self, i, j, o):
            if j > 0:
                return self.dMX_dt[i, j, o] == \
                       sum(self.ldot_t[j, k] * self.MX[i, k, o] for k in self.cp)
            else:
                return Constraint.Skip
        
        self.dvar_t_MX = Constraint(self.fe_t, self.cp, self.o, rule=_collocation_MX)
        
        
        def _continuity_MX(self, i, o):
            if i < nfe and nfe > 1:
                return self.MX[i + 1, 0, o] - sum(self.l1_t[j] * self.MX[i, j, o] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_MX = Expression(self.fe_t, self.o, rule=_continuity_MX)
        self.cp_MX = Constraint(self.fe_t, self.o, rule=lambda self,i,o:self.noisy_MX[i,o] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_MX(self, o):
            return self.MX[1, 0, o] - self.MX_ic[o]
        
        self.MX_ice = Expression(self.o, rule=_init_MX)
        self.MX_icc = Constraint(self.o, rule=lambda self, o: self.MX_ice[o] == 0.0)
     
        def _ode_Y(self,i,j):
            if j > 0:
                return self.dY_dt[i,j] == (self.kr[i,j,'t']*self.n_KOH*self.PO[i,j]*self.PO_scale*self.Vi[i,j]*self.Vi_scale - self.kr[i,j,'i']*self.U[i,j]*self.U_scale*self.PO[i,j]*self.PO_scale*self.Vi[i,j]*self.Vi_scale)/self.Y_scale
            else:
                return Constraint.Skip
        self.de_Y = Constraint(self.fe_t, self.cp, rule=_ode_Y)
        
        
        def _collocation_Y(self, i, j):
            if j > 0:
                return self.dY_dt[i, j] == \
                       sum(self.ldot_t[j, k] * self.Y[i, k] for k in self.cp)
            else:
                return Constraint.Skip
        
        self.dvar_t_Y = Constraint(self.fe_t, self.cp, rule=_collocation_Y)
        
        
        def _continuity_Y(self, i):
            if i < nfe and nfe > 1:
                return self.Y[i + 1, 0] - sum(self.l1_t[j] * self.Y[i, j] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_Y = Expression(self.fe_t, rule=_continuity_Y)
        self.cp_Y = Constraint(self.fe_t, rule=lambda self,i:self.noisy_Y[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_Y(self):
            return self.Y[1, 0] - self.Y_ic
        
        self.Y_ice = Expression(rule=_init_Y)
        self.Y_icc = Constraint(rule=lambda self: self.Y_ice == 0.0)
        
        def _ode_MY(self,i,j):
            if j > 0:   
                return self.dMY_dt[i,j] == (self.kr[i,j,'i']*self.U[i,j]*self.U_scale*self.PO[i,j]*self.PO_scale*self.Vi[i,j]*self.Vi_scale)/self.MY0_scale
            else:
                return Constraint.Skip
        self.de_MY = Constraint(self.fe_t, self.cp, rule=_ode_MY)
        
        def _collocation_MY(self, i, j):
            if j > 0:
                return self.dMY_dt[i, j] == \
                       sum(self.ldot_t[j, k] * self.MY[i, k] for k in self.cp)
            else:
                return Constraint.Skip
        
        self.dvar_t_MY = Constraint(self.fe_t, self.cp, rule=_collocation_MY)
        
        
        def _continuity_MY(self, i):
            if i < nfe and nfe > 1:
                return self.MY[i + 1, 0] - sum(self.l1_t[j] * self.MY[i, j] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_MY = Expression(self.fe_t, rule=_continuity_MY)
        self.cp_MY = Constraint(self.fe_t, rule=lambda self,i:self.noisy_MY[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_MY(self):
            return self.MY[1, 0] - self.MY_ic
        
        self.MY_ice = Expression(rule=_init_MY)
        self.MY_icc = Constraint(rule=lambda self: self.MY_ice == 0.0)
        
        # energy balance
        def _collocation_T_cw(self,i,j):
            if j > 0:
                #implicit systematic
                #return self.dT_cw_dt[i] == sum(self.ldot_t[j, k] * self.T_cw[i, k]*self.T_scale for k in self.cp)/self.tf
                #explicit tailored to piecewise affine
                return self.T_cw[i,j]*self.T_scale == self.T_cw[i,0]*self.T_scale + self.dT_cw_dt[i]*self.tf*self.fe_dist[i]*self.tau_i_t[j]
            else:   
                return Constraint.Skip

        self.dvar_t_T_cw = Constraint(self.fe_t, self.cp, rule = _collocation_T_cw)
        
        def _continuity_T_cw(self,i):
            if i < nfe and nfe > 1:
                return self.T_cw[i+1,0] - sum(self.l1_t[j] * self.T_cw[i, j] for j in self.cp)
            else:
                return Expression.Skip 

        self.noisy_T_cw = Expression(self.fe_t, rule=_continuity_T_cw)
        self.cp_T_cw = Constraint(self.fe_t, rule=lambda self,i:self.noisy_T_cw[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_T_cw(self):
            return self.T_cw[1,0] - self.T_cw_ic
        
        self.T_cw_ice = Expression(rule=_init_T_cw)
        self.T_cw_icc = Constraint(rule=lambda self: self.T_cw_ice == 0.0)
        
        def _ode_T(self,i,j):    
            if j > 0:
                #return self.dT_dt[i,j]*(self.m_tot[i,j]*self.m_tot_scale)*(self.bulk_cp_1 + self.bulk_cp_2*self.T[i,j]*self.T_scale) == (-self.F[i]*self.Hrxn['p']*self.monomer_cooling[i,j]*self.tf*self.fe_dist[i] + \
                #                          self.Hrxn['p']*(self.F[i]*self.tf*self.fe_dist[i] - self.dPO_dt[i,j]*self.PO_scale + self.dW_dt[i,j]*self.W_scale) - self.k_c*self.tf*self.fe_dist[i]*(self.T[i,j]*self.T_scale - self.T_cw[i]*self.T_scale))/self.T_scale
                return self.dT_dt[i,j]*(self.m_tot[i,j]*self.m_tot_scale*(self.bulk_cp_1 + self.bulk_cp_2*self.T[i,j]*self.T_scale))/self.Hrxn['p'] ==\
                            (self.Qr[i,j] - self.Qc[i,j] - self.F[i]*self.tf*self.fe_dist[i]*self.mw_PO*self.monomer_cooling[i,j]*self.monomer_cooling_scale)/self.T_scale
            else:
                return Constraint.Skip
    
        self.de_T = Constraint(self.fe_t, self.cp, rule=_ode_T)             
        
        def _collocation_T(self, i, j):
            if j > 0:
                return self.dT_dt[i, j] == sum(self.ldot_t[j, k] * self.T[i, k] for k in self.cp)
            else:
                return Constraint.Skip
            
        self.dvar_t_T = Constraint(self.fe_t, self.cp, rule=_collocation_T)
              
        def _continuity_T(self, i):  
            if i < nfe and nfe > 1:
                return self.T[i + 1, 0] - sum(self.l1_t[j] * self.T[i, j] for j in self.cp)
            else:
                return Expression.Skip
        
        self.noisy_T = Expression(self.fe_t, rule=_continuity_T)
        self.cp_T = Constraint(self.fe_t, rule=lambda self,i:self.noisy_T[i] == 0.0 if i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_T(self):        
            return self.T[1, 0] - self.T_ic
        
        self.T_ice = Expression(rule=_init_T)
        self.T_icc = Constraint(rule=lambda self: self.T_ice == 0.0)      
        
        # kinetics
        def _rxn_rate_r_a(self,i,j,r):
            if j > 0:
                return 0.0 == (self.T[i,j]*self.T_scale*log(self.A[r]*self.p_A[i,r]*self.scale*60*1000) - self.Ea[r]/self.Rg - self.T[i,j]*self.T_scale*self.k_l[i,j,r])
            else:
                return Constraint.Skip
            
        self.rxn_rate_r_a = Constraint(self.fe_t, self.cp, self.r, rule=_rxn_rate_r_a)
        
        def _rxn_rate_r_b(self,i,j,r):
            if j > 0:
                return 0.0 == exp(self.k_l[i,j,r])*self.tf*self.fe_dist[i] - self.kr[i,j,r]
            else:
                return Constraint.Skip
            
        self.rxn_rate_r_b = Constraint(self.fe_t, self.cp, self.r, rule=_rxn_rate_r_b)

        # algebraic equations
        def _ae_V(self,i,j):
            if j == 0:
                return Constraint.Skip
            else:
                return 0.0 == (1e3 - self.Vi[i,j]*self.Vi_scale * self.m_tot[i,j]*self.m_tot_scale*(1+0.0007576*(self.T[i,j]*self.T_scale - 298.15)))
        
        self.ae_V = Constraint(self.fe_t, self.cp, rule=_ae_V)
        
        def _ae_equilibrium_a(self,i,j):
            if j == 0:
                return Constraint.Skip
            else:
                return 0.0 == self.G[i,j]*self.G_scale*(self.MX[i,j,0]*self.MX0_scale + self.MY[i,j]*self.MY0_scale + self.X[i,j]*self.X_scale + self.Y[i,j]*self.Y_scale) - self.X[i,j]*self.X_scale*self.n_KOH
        
        self.ae_equilibrium_a = Constraint(self.fe_t, self.cp, rule =_ae_equilibrium_a)
        
        def _ae_equilibrium_b(self,i,j):
            if j == 0:
                return Constraint.Skip
            else:
                return 0.0 == self.U[i,j]*self.U_scale*(self.MX[i,j,0]*self.MX0_scale + self.MY[i,j]*self.MY0_scale + self.X[i,j]*self.X_scale + self.Y[i,j]*self.Y_scale) - self.Y[i,j]*self.Y_scale*self.n_KOH
        
        self.ae_equilibrium_b = Constraint(self.fe_t, self.cp, rule =_ae_equilibrium_b)
        
        def _ae_equilibrium_c(self,i,j):
            if j > 0:
                return 0.0 == self.MG[i,j]*(self.MX[i,j,0]*self.MX0_scale + self.MY[i,j]*self.MY0_scale + self.X[i,j]*self.X_scale + self.Y[i,j]*self.Y_scale) - self.MX[i,j,0]*self.MX0_scale*self.n_KOH
            else:
                return Constraint.Skip
                
        self.ae_equilibrium_c = Constraint(self.fe_t, self.cp, rule =_ae_equilibrium_c)
        
        # constraints
        def _pc_heat_a(self,i,j):
            if j == 0:
                return Constraint.Skip
            else:
                return 0.0 == self.bulk_cp_1*(self.T[i,j]*self.T_scale) + self.bulk_cp_2*(self.T[i,j]*self.T_scale)**2/2.0 - self.int_T[i,j]*self.int_T_scale
        
        self.pc_heat_a = Constraint(self.fe_t, self.cp, rule=_pc_heat_a)
        
        def _pc_heat_b(self,i,j):
            if j == 0:
                return Constraint.Skip
            else:
                return 0.0 == self.bulk_cp_1*self.Tad[i,j]*self.Tad_scale + self.bulk_cp_2*(self.Tad[i,j]*self.Tad_scale)**2/2.0 - self.int_Tad[i,j]*self.int_Tad_scale
        
        self.pc_heat_b = Constraint(self.fe_t, self.cp, rule=_pc_heat_b)
        
        def _Q_in(self,i,j):
            if j > 0: 
                return self.Qr[i,j] == ((self.kr[i,j,'i']-self.kr[i,j,'p'])*(self.G[i,j]*self.G_scale + self.U[i,j]*self.U_scale) + (self.kr[i,j,'p'] + self.kr[i,j,'t'])*self.n_KOH + self.kr[i,j,'a']*self.W[i,j]*self.W_scale)*self.PO[i,j]*self.PO_scale*self.Vi[i,j]*self.Vi_scale*self.Hrxn_aux['p']*self.p_Hrxn_aux[i,'p'] + self.dW_dt[i,j]*self.W_scale * self.Hrxn_aux['p'] * self.p_Hrxn_aux[i,'p']
            else:
                return Constraint.Skip
        
        self.Q_in = Constraint(self.fe_t, self.cp, rule=_Q_in)
        
        def _Q_out(self,i,j):
            if j > 0:
                return self.Qc[i,j] == self.kA*self.p_kA[i]*self.tf*self.fe_dist[i] * (self.T[i,j]*self.T_scale - self.T_cw[i,j]*self.T_scale)
            else:
                return Constraint.Skip
        
        self.Q_out = Constraint(self.fe_t, self.cp, rule=_Q_out)
        
        
        def _pc_temp_a(self,i,j):
            if j == 0:
                return Constraint.Skip
            else:
                return 0.0 == self.m_tot[i,j]*self.m_tot_scale*(self.int_Tad[i,j]*self.int_Tad_scale - self.int_T[i,j]*self.int_T_scale) - self.PO[i,j]*self.PO_scale*self.Hrxn['p']*self.Hrxn_aux['p']*self.p_Hrxn_aux[i,'p']
        
        self.pc_temp_a = Constraint(self.fe_t, self.cp, rule=_pc_temp_a)
        
        def _pc_temp_b(self,i,j):
            if j == 0:
                return Constraint.Skip
            else:
                return 0.0 == (self.T_safety + self.Tb) - self.Tad[i,j]*self.Tad_scale - self.s_temp_b[i,j] + self.eps_pc[i,j,1] - self.xi_temp_b[i,j]
            
        self.pc_temp_b = Constraint(self.fe_t, self.cp, rule = _pc_temp_b)   
        
        def _pc_T_max(self,i,j):
            if j > 0:
                return 0.0 == (self.T_max + self.Tb) - self.T[i,j]*self.T_scale - self.s_T_max[i,j] + self.eps_pc[i,j,2] - self.xi_T_max[i,j]
            else:
                return Constraint.Skip
        
        self.pc_T_max = Constraint(self.fe_t, self.cp, rule = _pc_T_max)
        
        def _pc_T_min(self,i,j):
            if j > 0:
                return 0.0 == self.T[i,j]*self.T_scale - (self.T_min + self.Tb) - self.s_T_min[i,j] + self.eps_pc[i,j,3] - self.xi_T_min[i,j]
            else:
                return Constraint.Skip
        
        self.pc_T_min = Constraint(self.fe_t, self.cp, rule = _pc_T_min)
        
        def _pc_heat_removal_c(self,i,j):
            if j == 0:
                return Constraint.Skip
            else:
                return 0.0 == self.mono_cp_1*(self.T[i,j]*self.T_scale - self.feed_temp) + self.mono_cp_2/2.0 *((self.T[i,j]*self.T_scale)**2.0-self.feed_temp**2.0) + self.mono_cp_3/3.0*((self.T[i,j]*self.T_scale)**3.0-self.feed_temp**3.0) + self.mono_cp_4/4.0*((self.T[i,j]*self.T_scale)**4.0-self.feed_temp**4.0) \
                              - self.Hrxn['p']*self.monomer_cooling[i,j]*self.monomer_cooling_scale
         
        self.pc_heat_removal_c = Constraint(self.fe_t, self.cp, rule=_pc_heat_removal_c)
        
        def _epc_PO_ptg(self,i,j):
            if i == nfe and j == ncp:
                return  0.0 == self.unreacted_PO - 1e6*self.PO[i,j]*self.PO_scale*self.mw_PO/(self.m_tot[i,j]*self.m_tot_scale) + self.eps[1] - self.s_PO_ptg - self.xi_PO_ptg
            else:
                return Constraint.Skip
        
        self.epc_PO_ptg = Constraint(self.fe_t, self.cp, rule=_epc_PO_ptg)    
        
        
        def _epc_unsat(self,i,j):
            if i == nfe and j == ncp:
                return  0.0 == self.unsat_value - 1000.0*(self.MY[i,j]*self.MY0_scale + self.Y[i,j]*self.Y_scale)/(self.m_tot[i,j]*self.m_tot_scale) + self.eps[2] - self.s_unsat - self.xi_unsat
            else: 
                return Constraint.Skip
        
        self.epc_unsat = Constraint(self.fe_t, self.cp, rule=_epc_unsat)
        
        def _epc_PO_fed(self,i,j):
            if i == nfe and j == ncp:
                return 0.0 == self.PO_fed[i,j]*self.PO_fed_scale - self.n_PO - self.s_PO_fed 
            else:
                return Constraint.Skip
        
        self.epc_PO_fed = Constraint(self.fe_t, self.cp, rule=_epc_PO_fed)
        
        def _epc_mw(self,i,j):
            if i == nfe and j == ncp:
                return  0.0 == self.MX[i,j,1]*self.MX1_scale/(self.MX[i,j,0]*self.MX0_scale)*self.mw_PO*self.num_OH + self.mw_PG.value - self.molecular_weight + self.eps[3] - self.s_mw - self.xi_mw
            else:
                return Constraint.Skip
        
        self.epc_mw = Constraint(self.fe_t, self.cp, rule=_epc_mw)
        
        def _epc_mw_ub(self):
            return 0.0 == -(self.MX[nfe,ncp,1]*self.MX1_scale - (self.molecular_weight_max - self.mw_PG)/self.mw_PO/self.num_OH*self.MX[nfe,ncp,0]*self.MX0_scale) + self.eps[4] - self.s_mw_ub - self.xi_mw_ub
        
        self.epc_mw_ub = Constraint(rule =_epc_mw_ub)
        
        # controls (technicalities)
        self.u1_e = Expression(self.fe_t, rule = lambda self, i: self.dT_cw_dt[i])
        self.u2_e = Expression(self.fe_t, rule = lambda self, i: self.F[i])
        self.u1_c = Constraint(self.fe_t, rule = lambda self, i: self.u1_e[i] == self.u1[i])
        self.u2_c = Constraint(self.fe_t, rule = lambda self, i: self.u2_e[i] == self.u2[i])
        
        # objective
        def _obj(self):
            return self.tf + self.rho*(sum(self.eps[i] for i in self.epc) + sum(sum(sum(self.eps_pc[i,j,k] for i in self.fe_t) for j in self.cp if j > 0) for k in self.pc))
#            return self.tf + self.rho*(sum(self.eps[i] for i in self.epc) + sum(sum(sum(self.eps_pc[i,j,k] for i in self.fe_t) for j in self.cp if j > 0) for k in self.pc)) \
#                    + self.gamma * (self.MX[self.nfe,self.ncp,1]*self.MX1_scale/(self.MX[self.nfe,self.ncp,0]*self.MX0_scale)*self.mw_PO*self.num_OH + self.mw_PG - self.molecular_weight)**2
#        self.epc_mw_ub.deactivate()
#        self.epc_mw.deactivate()
        
        self.eobj = Objective(rule=_obj,sense=minimize)
        
        
        #Suffixes
        self.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)                  
             

    def e_state_relation(self):
        # uses implicit assumption of 
        self.add_component('W_e', Var(initialize=self.W[self.nfe,self.ncp].value))
        self.add_component('PO_e', Var(initialize=self.PO[self.nfe,self.ncp].value))
        self.add_component('T_e', Var(initialize=self.T[self.nfe,self.ncp].value))
        self.add_component('Y_e', Var(initialize=self.Y[self.nfe,self.ncp].value))
        self.add_component('MY_e', Var(initialize=self.MY[self.nfe,self.ncp].value))
        self.add_component('MX_e', Var(self.o, initialize={0:self.MX[self.nfe,self.ncp,0].value,1:self.MX[self.nfe,self.ncp,1].value}))
        self.add_component('PO_fed_e', Var(initialize=self.W[self.nfe,self.ncp].value))
        self.add_component('T_cw_e', Var(initialize=self.T_cw[self.nfe,self.ncp].value))
        
        self.add_component('W_e_expr', Expression(rule=lambda self: self.W_e - self.W[self.nfe,self.ncp]))
        self.add_component('PO_e_expr', Expression(rule=lambda self: self.PO_e - self.PO[self.nfe,self.ncp]))
        self.add_component('T_e_expr', Expression(rule=lambda self: self.T_e - self.T[self.nfe,self.ncp]))
        self.add_component('Y_e_expr', Expression(rule=lambda self: self.Y_e - self.Y[self.nfe,self.ncp]))
        self.add_component('MY_e_expr', Expression(rule=lambda self: self.MY_e - self.MY[self.nfe,self.ncp]))
        self.add_component('MX_e_expr', Expression(self.o, rule=lambda self,o: self.MX_e[o] - self.MX[self.nfe,self.ncp,o]))
        self.add_component('PO_fed_e_expr', Expression(rule=lambda self: self.PO_fed_e - self.PO_fed[self.nfe,self.ncp])) 
        self.add_component('T_cw_e_expr', Expression(rule=lambda self: self.T_cw_e - self.T_cw[self.nfe,self.ncp]))
        
        self.W_e_c = Constraint(rule=lambda self: self.W_e_expr == 0.0)
        self.PO_e_c = Constraint(rule=lambda self: self.PO_e_expr == 0.0)
        self.T_e_c = Constraint(rule=lambda self: self.T_e_expr == 0.0)
        self.Y_e_c = Constraint(rule=lambda self: self.Y_e_expr == 0.0)
        self.MY_e_c = Constraint(rule=lambda self: self.MY_e_expr == 0.0)
        self.MX_e_c = Constraint(self.o, rule=lambda self,o: self.MX_e_expr[o] == 0.0)
        self.PO_fed_e_c = Constraint(rule=lambda self: self.PO_fed_e_expr == 0.0)
        self.T_cw_e_c = Constraint(rule=lambda self: self.T_cw_e_expr == 0.0)
        
    def par_to_var(self):
        self.A['i'].setlb(396400.0*0.5/self.scale)
        self.A['i'].setub(396400.0*1.5/self.scale)
        #self.A['i'].value = 396400.0/self.scale
        
        self.A['p'].setlb(13504.2*0.5/self.scale)
        self.A['p'].setub(13504.2*1.5/self.scale)
        #self.A['p'].value = 13504.2/self.scale
        
        self.A['t'].setlb(1.509e6*0.5/self.scale)
        self.A['t'].setub(1.509e6*1.5/self.scale)
        #self.A['t'].value = 1.509e6/self.scale
        
        self.Hrxn_aux['p'].setlb(0.5)
        self.Hrxn_aux['p'].setlb(1.5)
        #self.Hrxn_aux['p'].value = 1.0
        
        self.kA.setlb(0.5*2200.0/self.Hrxn['p']*60/20.0)
        self.kA.setub(1.5*2200.0/self.Hrxn['p']*60/20.0)
        


        #self.kA.value = 2200.0/self.Hrxn['p']*60/20.0
    def create_output_relations(self):
        self.add_component('MW', Var(self.fe_t,self.cp, initialize=0.0, bounds=(0,None)))
        self.add_component('MW_c', Constraint(self.fe_t, self.cp))            
        self.MW_c.rule = lambda self, i, j: self.MX[i,j,1]*self.MX1_scale - (self.MW[i,j]*self.MW_scale - self.mw_PG)/self.mw_PO/self.num_OH*self.MX[i,j,0]*self.MX0_scale == 0.0 if j > 0 else Constraint.Skip
        self.MW_c.reconstruct()
        
        self.add_component('ByProd', Var(self.fe_t, self.cp, initialize=0.0, bounds=(0,None)))
        self.add_component('ByProd_c', Constraint(self.fe_t, self.cp))
        self.ByProd_c.rule = lambda self, i, j:  self.MY[i,j] + self.Y[i,j] - self.ByProd[i,j] == 0.0 if j > 0 else Constraint.Skip
        self.ByProd_c.reconstruct()
        
#        self.add_component('Prod', Var(self.fe_t, self.cp, initialize=0.0, bounds=(0,None)))
#        self.add_component('Prod_c', Constraint(self.fe_t, self.cp))
#        self.Prod_c.rule = lambda self, i, j:  self.MX[i,j,0] + self.X[i,j] - self.Prod[i,j] == 0.0 if j > 0 else Constraint.Skip
#        self.Prod_c.reconstruct()
        
    def equalize_u(self, direction="u_to_r"):
        """set current controls to the values of their respective dummies"""
        if direction == "u_to_r":
            for i in iterkeys(self.dT_cw_dt):
                self.dT_cw_dt[i].set_value(value(self.u1[i]))
            for i in iterkeys(self.F):
                self.F[i].set_value(value(self.u2[i]))
        elif direction == "r_to_u":
            for i in iterkeys(self.u1):
                self.u1[i].value = value(self.dT_cw_dt[i])
            for i in iterkeys(self.u2):
                self.u2[i].value = value(self.F[i])    
    
    def deactivate_epc(self):
        self.epc_PO_fed.deactivate()
        self.epc_PO_ptg.deactivate()
        self.epc_unsat.deactivate()
        self.epc_mw.deactivate()
        self.epc_mw_ub.deactivate() 
        
    def deactivate_pc(self):
        self.pc_temp_b.deactivate()     
        self.pc_T_max.deactivate()
        self.pc_T_min.deactivate()
    
    def clear_all_bounds(self):
        for var in self.component_objects(Var):
            for key in var.index_set():
                var[key].setlb(None)
                var[key].setub(None)
    
    def clear_aux_bounds(self):
        keep_bounds = ['s_temp_b','s_T_min','s_T_max','s_mw','s_PO_ptg','s_unsat','s_mw','s_mw_ub','s_PO_fed','eps','eps_pc','F','u1','u2','tf','k_l','T_cw','T'] 
        for var in self.component_objects(Var, active=True):
            if var.name in keep_bounds:
                continue
            else:
                for key in var.index_set():
                    var[key].setlb(None)
                    var[key].setub(None)
    
    def clear_aux_bounds_mhe(self):
        keep_bounds = ['s_temp_b','s_T_min','s_T_max','s_mw','s_PO_ptg','s_unsat','s_mw','s_mw_ub','s_PO_fed','eps','eps_pc','F','u1','u2','tf','k_l'] 
        for var in self.component_objects(Var, active=True):
            if var.name in keep_bounds:
                continue
            else:
                for key in var.index_set():
                    var[key].setlb(None)
                    var[key].setub(None)        
                           
    def create_bounds(self):
        self.tf.setlb(min(10.0,10.0*24.0/self.nfe))
        self.tf.setub(min(20.0,20.0*24/self.nfe))
        for i in self.fe_t:
            self.dT_cw_dt[i].setlb(-50.0/10.0)
            self.u1[i].setlb(-50.0/10.0)
            self.dT_cw_dt[i].setub(50.0/10.0)
            self.u1[i].setub(50.0/10.0)
            self.F[i].setlb(0.0)
            self.u2[i].setlb(0.0)
            self.F[i].setub(3.0) 
            self.u2[i].setub(3.0)
            for j in self.cp:
                self.T_cw[i,j].setlb(293.15/self.T_scale)
                self.T_cw[i,j].setub((self.T_max + self.Tb)/self.T_scale)
                self.T[i,j].setlb((50.0 + self.Tb)/self.T_scale)
                self.T[i,j].setub((200.0 + self.Tb)/self.T_scale)
                self.int_T[i,j].setlb((1.1*(100+self.Tb) + 2.72*(100+self.Tb)**2/2000)/self.int_T_scale)
                self.int_T[i,j].setub((1.1*(170+self.Tb) + 2.72*(170+self.Tb)**2/2000)/self.int_T_scale)
                self.Vi[i,j].setlb(0.9/self.Vi_scale*(1e3)/((self.m_KOH + self.m_PG + self.m_PO + self.m_H2O)*(1 + 0.0007576*((170+self.Tb)-298.15))))
                self.Vi[i,j].setub(1.1/self.Vi_scale*(1e3)/((self.m_KOH + self.m_PG + self.m_H2O)*(1 + 0.0007576*((100+self.Tb)-298.15))))
                self.Tad[i,j].setlb((100 + self.Tb)/self.Tad_scale)
                self.Tad[i,j].setub((self.T_safety + self.Tb)/self.Tad_scale)
                for r in self.r:
                    self.k_l[i,j,r].setlb(None)
                    self.k_l[i,j,r].setub(20.0)

    def del_pc_bounds(self):
        for i in self.fe_t:
            for j in self.cp:
                self.int_T[i,j].setlb(None)
                self.int_T[i,j].setub(None)
                self.Tad[i,j].setlb(None)
                self.Tad[i,j].setub(None)
                self.Vi[i,j].setlb(0)
                self.Vi[i,j].setub(None)
                

    def write_nl(self):
        """Writes the nl file and the respective row & col"""
        name = str(self.__class__.__name__) + ".nl"
        self.write(filename=name,
                   format=ProblemFormat.nl,
                   io_options={"symbolic_solver_labels": True})
        
   
    def initialize_element_by_element(self):
        print('initializing element by element ...')
        m_aux = SemiBatchPolymerization(1,self.ncp)
        m_aux.eobj.deactivate()
        m_aux.deactivate_epc()        
        m_aux.deactivate_pc()
        m_aux.F[1] = 1.2
        m_aux.dvar_t_T_cw.deactivate()
        m_aux.T_cw_icc.deactivate()
        m_aux.T_cw.fix(397.0/self.T_scale)
        m_aux.tf = min(12*24.0/self.nfe,12)
        m_aux.F[1].fixed = True
        m_aux.tf.fixed = True
        opt = SolverFactory('ipopt')
        opt.options["halt_on_ampl_error"] = "yes"
        opt.options["max_iter"] = 5000
        opt.options["tol"] = 1e-5
        results = {}
        k = 0
        # solve square problem
        for fe_t in self.fe_t:
            results[fe_t] = m_aux.save_results(opt.solve(m_aux, tee=False, keepfiles=False))
            m_aux.troubleshooting()
            prevsol = results[fe_t]
            try:
                m_aux.T_ic = prevsol['T',(1,3)]
                m_aux.MY_ic = prevsol['MY',(1,3)]
                m_aux.W_ic = prevsol['W',(1,3)]
                m_aux.PO_ic = prevsol['PO',(1,3)]
                m_aux.Y_ic = prevsol['Y',(1,3)]
                m_aux.PO_fed_ic = prevsol['PO_fed',(1,3)]
                m_aux.T_cw_ic = prevsol['T_cw',(1,3)]
                #m_aux.m_tot_ic = prevsol['m_tot',(1,3)]
                #m_aux.X_ic = prevsol['X',(1,3)]
                # initial guess for next element
                for i in m_aux.fe_t:
                    for j in m_aux.cp:
                        m_aux.T[i,j] = prevsol['T',(1,3)]
                        m_aux.MY[i,j] = prevsol['MY',(1,3)]
                        m_aux.W[i,j] = prevsol['W',(1,3)]
                        m_aux.PO[i,j] = prevsol['PO',(1,3)]
                        m_aux.m_tot[i,j] = prevsol['m_tot',(1,3)]
                        m_aux.X[i,j] = prevsol['X',(1,3)]
                        m_aux.Y[i,j] = prevsol['Y',(1,3)]
                        m_aux.PO_fed[i,j] = prevsol['PO_fed',(1,3)]
                        m_aux.T_cw[i,j] = prevsol['T_cw',(1,3)]
                # initial values for next element
                for o in m_aux.o:
                    m_aux.MX_ic[o] = prevsol['MX',(1,3,o)]
                    for i in m_aux.fe_t:
                        for j in m_aux.cp:
                            m_aux.MX[i,j,o] = prevsol['MX',(1,3,o)]
            except KeyError:
                print('something went wrong during shifting element')
                break
            if results[fe_t]['solstat'] == ['ok','optimal']:
                print('----> element %i converged' % fe_t)
            else:
                return m_aux
                break
            k += 1
        
        if k == self.nfe:
            # load results into optimization model 'self'
            # results[number of finite element] = {'VarName',(index tuple):value}
            for var in self.component_objects(Var, active=True):
                i=0
                aux_index_set = list(var.index_set())
                for key in var.index_set():
                    # keys in model are arranged in the following way:
                    # key[0] = number of finite element
                    # key[1] = number of collocation point
                    # key[1:] = indeces from remaining index sets
                    if aux_index_set[i] == None or type(key)==str:
                        # non-index variable (only m.tf --> already initialized in real model)
                        break
                    elif isinstance(aux_index_set[i],collections.Sequence): # only one index
                        aux_key = list(aux_index_set[i])
                        fe_t = aux_key[0]
                        aux_key[0] = 1
                        aux_key = tuple(aux_key)
                        if var.name != 'MW':
                            var[key] = results[fe_t][var.name,aux_key]
                    else:
                        pass
                    #else: # multiple indices
                    #    aux_key = 1
                    #    var[key] = results[i+1][var.name,aux_key]
                    i+=1
            print('...initialization complete')
        else:
            print('...initialization failed')
                
    def save_results(self,solution):
        # solution is output from solver.solve(model)
        output = {}
        for _var in self.component_objects(Var, active=True):
            for _key in _var.index_set():
                try:
                    output[_var.name,_key] = _var[_key].value
                except KeyError:
                    continue
        output['solstat'] = [str(solution.solver.status), str(solution.solver.termination_condition)]
        return output       
        
    def plot_profiles(self, v,**kwargs):
        #v = list of variables for which the profiles are plotted
        #only works for properties with 1 or two indices so far 
        #thus can not plot moments like this
        for k in range(len(v)):
            values = []
            t0 = kwargs.pop('t0', 0)
            t = []
            i = 0
            aux = list(v[k].index_set())
            for key in v[k].index_set():         
                if v[k][key].value == None:
                    continue
                else:
                    # check if piecewise constant or continuous profile
                    if isinstance(aux[i],collections.Sequence):
                        # continuous
                        l = list(aux[i])[0] # finite element index 
                        p = list(aux[i])[1] # collocation point index
                        if not(p==0):
                            t.append((l+self.tau_i_t[p])*self.tf.value)
                            values.append(v[k][key].value)
                    else: 
                        # piecewise constant
                        t.append(i*self.tf.value)
                        t.append((i+1)*self.tf.value)
                        values.append(v[k][key].value)
                        values.append(v[k][key].value)
                    i+=1
            values = np.array(values)
            t = np.array(t)
            t = t+t0
            plt.figure(k)
            plt.plot(t,values)
            plt.title(v[k].name + '-profile')
            plt.ylabel(v[k].name)
            plt.xlabel('$t \, [min]$')
                        
    def check_feasibility(self, display = False):
        # evaluates the rhs for every endpoint constraint
        epsilon = {}
        i = self.nfe
        j = self.ncp  # assmuning radau nodes
        epsilon['epc_PO_ptg'] = self.unreacted_PO.value - self.PO[i,j].value*self.PO_scale*self.mw_PO.value/(self.m_tot[i,j].value*self.m_tot_scale)*1e6# 
        epsilon['epc_mw'] = self.MX[i,j,1].value*self.MX1_scale/(self.MX[i,j,0].value*self.MX0_scale)*self.mw_PO.value*self.num_OH.value + self.mw_PG.value - self.molecular_weight.value 
        epsilon['epc_unsat'] = self.unsat_value.value - 1000.0*(self.MY[i,j].value*self.MY0_scale + self.Y[i,j].value*self.Y_scale)/(self.m_tot[i,j].value*self.m_tot_scale)
        
        if display:
            print(epsilon)
        return epsilon
    
    def get_NAMW(self):
        i = self.nfe
        j = self.ncp
        return self.mw_PG.value + self.mw_PO.value*self.num_OH.value*self.MX[i,j,1].value/self.MX[i,j,0].value
    
    def troubleshooting(self):
        with open("troubleshooting.txt", "w") as f:
            self.display(ostream=f)
            f.close()
    
        with open("pprint.txt","w") as f:
            self.pprint(ostream=f)
            f.close()


###############################################################################
#######################            Test Run             #######################
###############################################################################
#
#Solver = SolverFactory('ipopt')
#Solver.options["halt_on_ampl_error"] = "yes"
#Solver.options["max_iter"] = 5000
#Solver.options["tol"] = 1e-8
#Solver.options["linear_solver"] = "ma57"
#f = open("ipopt.opt", "w")
#f.write("print_info_string yes")
#f.close()
#
#e = SemiBatchPolymerization(24,3)
#m = e.initialize_element_by_element()
#e.create_output_relations()
#e.create_bounds()
#e.clear_aux_bounds()
#
##e.T_icc.deactivate()
##e.T_cw_icc.deactivate()
##e.T_icc.deactivate()
##e.T[1,0].setlb(373.15/e.T_scale)
##e.T_cw_icc.deactivate()
##for index in e.F.index_set():
##    e.F[index].setlb(0)
##    e.F[index].setub(4.0)
#res=Solver.solve(e,tee=True)
#e.plot_profiles([e.PO,e.W,e.Y,e.PO_fed,e.MY,e.T,e.T_cw,e.F,e.dT_cw_dt,e.Tad])
#h = e.save_results(res)
#f = open('optimal_trajectory.pckl','wb')
#pickle.dump(h,f)
#f.close()


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
from pyomo.dae import *
from pyomo.opt import ProblemFormat
from aux.cpoinsc import collptsgen
from aux.lagrange_f import lgr, lgry, lgrdot, lgrydot
from six import itervalues, iterkeys, iteritems
import sys, collections
import matplotlib.pyplot as plt
import numpy as np


class SemiBatchPolymerization_multistage(ConcreteModel):
    def __init__(self, nfe, ncp, **kwargs):   
        ConcreteModel.__init__(self)        
        # required arguments
        self.nfe = nfe # number of finite elements
        self.ncp = ncp # number of collocation points
        
        # keyboardarguments
        self.s_max = kwargs.pop('s_max',1) # number of scenarios
        self.nr = kwargs.pop('robust_horizon', 1) # robust horizon
        dummy_tree = {}
        for i in range(1,self.nfe+1):
            dummy_tree[i,1] = (i-1,1,1,{('A',('p',)):1.0,('A',('i',)):1.0})
        self.scenario_tree = kwargs.pop('scenario_tree',dummy_tree) # default should be a symmetric scenario tree
        
        # scaling factors
        self.W_scale = 1
        self.Y_scale = 1e-2
        self.PO_scale = 1e1
        self.MY0_scale = 1e-1
        self.MX0_scale = 1e1
        self.MX1_scale = 1e2
        self.MW_scale = 1e2
        self.X_scale = 1
        self.m_tot_scale = 1e4
        self.T_scale = 1e2
        self.Tad_scale = 1e2
        self.Vi_scale = 1e-2
        self.PO_fed_scale = 1e2
        self.int_T_scale = 1e2
        self.int_Tad_scale = 1e2
        self.G_scale = 1
        self.U_scale = 1e-2
        self.monomer_cooling_scale = 1e-2
        self.scale = 1.0
        
        # collocation pts
        self.tau_t = collptsgen(ncp, 1, 0) #compute normalized lagrange interpolation polynomial values according to desired collocation scheme
        
        # start at zero
        self.tau_i_t = {0: 0.}
        
        # create a list
        for ii in range(1, ncp + 1):
            self.tau_i_t[ii] = self.tau_t[ii - 1]
    
        # sets:
        self.s = Set(initialize=[i for i in range(1,self.s_max+1)])
        self.o = Set(initialize=[i for i in range(2)]) # moments
        self.r = Set(initialize=['a','i','p','t']) # reactions
        self.fe_t = Set(initialize=[i for i in range(1,nfe+1)])
        self.cp = Set(initialize=[i for i in range(ncp+1)])
        self.epc = Set(initialize=[i for i in range(1,5)])
        self.pc = Set(initialize=[i for i in range(1,4)])
 
        # time horizon
        self.tf = Var(self.fe_t, self.s, initialize=9.6*60/nfe,bounds=(2*60,9.20*60)) # batch time in [min]
        
        # parameter for different models
        self.p_A = Param(self.r, self.fe_t, self.s, initialize=1.0, mutable=True)
        self.p_Hrxn_aux = Param(self.r, self.fe_t, self.s, initialize=1.0, mutable = True)
        self.p_kA = Param(self.fe_t, self.s, initialize=1.0, mutable=True)
        self.p_n_KOH = Param(self.fe_t, self.s, initialize=1.0, mutable=True)
        # set parameter values
        self.set_scenarios()
        
        # parameters for l1-relaxation of endpoint-constraints
        self.eps = Var(self.epc, self.s, initialize=0, bounds=(0,None))
        self.eps.fix()
        self.eps_pc = Var(self.fe_t, self.cp, self.pc, self.s, initialize=0.0, bounds=(0,None))
        self.rho = Param(initialize=1e3, mutable=True)
        self.gamma = Param(initialize=10.0, mutable=True)
        
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
        self.n_KOH.fix()
        self.n_PG = Param(initialize=self.m_PG/self.mw_PG) # [kmol] mole of PG;
        
        # reactor and product specs
        self.T_safety = Param(initialize=170.0) #190.0 [°C] maximum allowed temperature after adiabatic temperature rise
        self.T_max = Param(initialize=150.0)
        self.T_min = Param(initialize=100.0)
        self.molecular_weight = Param(initialize=949.5, mutable=True) # 3027.74 # [g/mol] or [kg/kmol] target molecular weights
        self.molecular_weight_max = Param(initialize=949.5+20, mutable=True)
        self.unsat_value = Param(initialize=0.032) #0.032 # unsaturation value
        self.unreacted_PO = Param(initialize=120.0) #120.0 # [PPM] unreacted PO
        self.rxr_volume = Param(initialize=41.57) # [m^3] volume of the reactor
        self.rxr_pressure = Param(initialize=253) # [kPa] initial pressure
        self.rxr_temperature = Param(initialize=122.9) # [°C] initial temperature 
        self.feed_temp = Param(initialize=25+self.Tb) # [K] feed temperature of monomers
        
        # polymerization kinetics
        aux = np.array([8.64e4,3.964e5,1.35042e4,1.509e6]) # [m^3/mol/s]
        self.A = Var(self.r,initialize=({'a':aux[0]/self.scale,'i':aux[1]/self.scale,'p':aux[2]/self.scale,'t':aux[3]/self.scale}), bounds = (1e3,1e8)) #
        self.A.fix()
        self.Ea = Param(self.r,initialize=({'a':82.425,'i':77.822,'p':69.172,'t':105.018}), mutable=True) # [kJ/mol] activation engergy 
        self.Hrxn = Param(self.r, initialize=({'a':0, 'i':92048, 'p':92048,'t':0}), mutable=True)
        self.Hrxn_aux = Var(self.r, initialize=({'a':1.0, 'i':1.0, 'p':1.0,'t':1.0})) 
        # Hrxn_aux: USED FOR ON-LINE ESTIMATION since everything is normalized to nominal value of Hrxn
        self.Hrxn_aux.fix()
        self.max_heat_removal = Param(initialize=2.2e3/self.Hrxn['p']*60, mutable=True) # 2.2e3/self.Hrxn['p']*60 [kmol (PO)/min] maximum amount of heat removal rate scaled by Hrxn('p') (see below)s
        
        # heat transfer
        self.kA = Var(initialize=2200.0/self.Hrxn['p']*60/20.0) 
        self.kA.fix()
        
        # parameters for initializing differential variabales
        self.W_ic = Param(initialize= self.n_H2O/self.W_scale, mutable=True)
        self.PO_ic = Param(initialize = 0.0, mutable=True)
        self.m_tot_ic = Param(initialize = (self.m_PG+self.m_KOH+self.m_H2O)/self.m_tot_scale, mutable=True)
        self.X_ic = Param(initialize = (self.n_PG*self.num_OH+self.n_H2O*self.num_OH)/self.X_scale, mutable=True)
        self.Y_ic = Param(initialize = 0.0, mutable=True)
        self.MY_ic = Param(initialize = 0.0, mutable=True)
        self.MX_ic = Param(self.o, initialize = 0.0, mutable=True)
        self.PO_fed_ic = Param(initialize = 0.0, mutable=True) 
        self.T_ic = Param(initialize=393.15/self.T_scale, mutable=True)
        self.T_cw_ic = Param(initialize = 373.15/self.T_scale, mutable=True) 
        
        # variables
        # decision variables/controls (piece wise constant)
        # piecewise constant controls
        self.F = Var(self.fe_t, self.s, initialize=1,bounds=(0.0,None))
        self.dT_cw_dt = Var(self.fe_t, self.s, initialize=0.0)
        self.u1 = Var(self.fe_t, self.s, initialize=397.0/self.T_scale)
        self.u2 = Var(self.fe_t, self.s, initialize=1,bounds=(0.0,None))
        
        # differential variables
        self.T_cw = Var(self.fe_t, self.cp, self.s, initialize=373.15/self.T_scale, bounds=(0.0,None))

        #
        self.T = Var(self.fe_t, self.cp, self.s, initialize=397.0/self.T_scale) # temperature
        self.dT_dt = Var(self.fe_t, self.cp, self.s, initialize=0.0)
        #
        self.W = Var(self.fe_t, self.cp, self.s, initialize=self.W_ic, bounds=(0.0,None)) # Water
        self.dW_dt = Var(self.fe_t, self.cp, self.s)
        
        self.PO = Var(self.fe_t, self.cp, self.s, initialize=self.PO_ic, bounds=(0.0,None)) # propylene oxide
        self.dPO_dt = Var(self.fe_t,self.cp, self.s)

        self.m_tot = Var(self.fe_t, self.cp, self.s, initialize=self.m_tot_ic, bounds=(0,None)) # total mass
        self.dm_tot_dt = Var(self.fe_t, self.cp, self.s)

        self.X = Var(self.fe_t, self.cp, self.s, initialize=self.X_ic, bounds=(0.0,None)) # saturated product
        self.dX_dt = Var(self.fe_t, self.cp, self.s)

        self.MX = Var(self.fe_t,self.cp,self.o, self.s, bounds=(0.0,None)) # moments of X
        self.dMX_dt = Var(self.fe_t,self.cp,self.o, self.s)

        self.Y = Var(self.fe_t,self.cp, self.s, initialize=self.Y_ic, bounds=(0.0,None)) # unsaturated by-product
        self.dY_dt = Var(self.fe_t,self.cp, self.s)

        self.MY = Var(self.fe_t,self.cp, self.s, initialize=self.MY_ic, bounds=(0.0,None)) # moments of Y
        self.dMY_dt = Var(self.fe_t,self.cp, self.s)

        self.PO_fed = Var(self.fe_t,self.cp,self.s,initialize=self.PO_fed_ic, bounds=(0.0,None)) # total amount of PO fed
        self.dPO_fed_dt = Var(self.fe_t,self.cp,self.s)

        
        # reactions
        self.k_l = Var(self.fe_t, self.cp, self.r, self.s, bounds=(None,20)) # rate coefficients
        self.kr = Var(self.fe_t,self.cp,self.r, self.s) # rate coefficients
        
        # algebraic variables
        self.Vi = Var(self.fe_t,self.cp,self.s, bounds=(0,1e10)) # current volume 
        self.G = Var(self.fe_t,self.cp,self.s) # 
        self.U = Var(self.fe_t,self.cp,self.s) # heat transfer coefficient 
        self.MG = Var(self.fe_t,self.cp,self.s) # 
        
        # thermodynamic
        self.int_T = Var(self.fe_t, self.cp, self.s, initialize=653/self.int_T_scale) # internal energy
        self.int_Tad = Var(self.fe_t, self.cp, self.s, initialize=700/self.int_Tad_scale) # internal energy after adiabatic depletion of monomer
        self.Tad = Var(self.fe_t, self.cp, self.s, initialize=417.7/self.Tad_scale) 
        self.heat_removal = Var(self.fe_t, self.cp, self.s, initialize=1.121) 
        self.monomer_cooling = Var(self.fe_t, self.cp, self.s, initialize=0.08) 
        self.Qr = Var(self.fe_t,self.cp, self.s, initialize=0.0)
        self.Qc = Var(self.fe_t,self.cp, self.s, initialize=0.0)
        
        # define slack variables
        # path constraints
        self.s_temp_b = Var(self.fe_t, self.cp, self.s, initialize = 0, bounds=(0,None))
        self.s_T_max = Var(self.fe_t, self.cp, self.s, initialize=0.0, bounds=(0,None))
        self.s_T_min = Var(self.fe_t, self.cp, self.s, initialize=0.0, bounds=(0,None))
        
        # endpoint constraints
        self.s_mw = Var(self.s, initialize=0, bounds=(0,None))
        self.s_PO_ptg = Var(self.s, initialize=0, bounds=(0,None))
        self.s_unsat = Var(self.s, initialize=0, bounds=(0,None))
        self.s_PO_fed = Var(self.s, initialize=0, bounds=(0,None))
        self.s_mw_ub = Var(self.s, initialize=0, bounds=(0,None))
        
        
        
        # closures
        def _total_mass_balance(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.m_tot[i,j,s]*self.m_tot_scale == self.PO_fed[i,j,s]*self.PO_fed_scale*self.mw_PO+self.m_tot_ic*self.m_tot_scale
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.total_mass_balance = Constraint(self.fe_t, self.cp, self.s, rule=_total_mass_balance)
        
        def _rxn_invariant(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return 10.0432852386*2 + 47.7348816877 == self.W[i,j,s]*2*self.W_scale + self.X[i,j,s]*self.X_scale + self.MX[i,j,0,s]*self.MX0_scale
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.rxn_invariant = Constraint(self.fe_t,self.cp,self.s,rule=_rxn_invariant)
        
        # system dynamics
        # Water
        def _ode_W(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dW_dt[i,j,s] == -self.kr[i,j,'a',s]*self.W[i,j,s]*self.PO[i,j,s]*self.PO_scale*self.Vi[i,j,s]*self.Vi_scale
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.de_W = Constraint(self.fe_t, self.cp, self.s, rule=_ode_W)
        
        
        def _collocation_W(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dW_dt[i,j,s] == \
                           sum(self.ldot_t[j, k] * self.W[i,k,s] for k in self.cp)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.dvar_t_W = Constraint(self.fe_t, self.cp, self.s, rule=_collocation_W)
        
        def _continuity_W(self,i,s):
            if (i+1,s) in self.scenario_tree:
                if i < nfe and nfe > 1:
                    parent_s = self.scenario_tree[(i+1,s)][1] # parent scenario
                    return self.W[i+1,0,s] - sum(self.l1_t[j] * self.W[i,j,parent_s] for j in self.cp)
                else:
                    return Expression.Skip
            else:
                return Expression.Skip
            
        self.noisy_W = Expression(self.fe_t, self.s, rule=_continuity_W)
        self.cp_W = Constraint(self.fe_t, self.s, rule=lambda self,i,s:self.noisy_W[i,s] == 0.0 if (i+1,s) in self.scenario_tree and i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_W(self,s):
            if (1,s) in self.scenario_tree: 
                if s == 1:
                    return self.W[1,0,s] - self.W_ic
                else:
                    return self.W[1,0,s] - self.W[1,0,1]
            else:
                return Expression.Skip
            
        self.W_ice = Expression(self.s, rule=_init_W)
        self.W_icc = Constraint(self.s, rule=lambda self,s: self.W_ice[s] == 0.0 if (1,s) in self.scenario_tree else Constraint.Skip)
        
        def _ode_PO(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dPO_dt[i,j,s] == (self.F[i,s]*self.tf[i,s]*self.fe_dist[i] - (((self.kr[i,j,'i',s]-self.kr[i,j,'p',s])*(self.G[i,j,s]*self.G_scale + self.U[i,j,s]*self.U_scale) + (self.kr[i,j,'p',s] + self.kr[i,j,'t',s])*self.n_KOH*self.p_n_KOH[i,s] + self.kr[i,j,'a',s]*self.W[i,j,s]*self.W_scale)*self.PO[i,j,s]*self.PO_scale*self.Vi[i,j,s]*self.Vi_scale))/self.PO_scale
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.de_PO = Constraint(self.fe_t, self.cp, self.s, rule=_ode_PO)
          
        def _collocation_PO(self,i,j,s):  
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dPO_dt[i,j,s] == \
                           sum(self.ldot_t[j,k] * self.PO[i,k,s] for k in self.cp)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.dvar_t_PO = Constraint(self.fe_t, self.cp, self.s, rule=_collocation_PO)
            
        def _continuity_PO(self,i,s):
            if (i+1,s) in self.scenario_tree:
                if i < nfe and nfe > 1:
                    parent_s = self.scenario_tree[(i+1,s)][1] # parent scenario
                    return self.PO[i+1,0,s] - sum(self.l1_t[j] * self.PO[i,j,parent_s] for j in self.cp)
                else:
                    return Expression.Skip
            else:
                return Expression.Skip
            
        self.noisy_PO = Expression(self.fe_t, self.s, rule=_continuity_PO)
        self.cp_PO = Constraint(self.fe_t, self.s, rule=lambda self,i,s: self.noisy_PO[i,s] == 0.0 if (i+1,s) in self.scenario_tree and i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_PO(self,s):
            if (1,s) in self.scenario_tree:
                if s == 1:
                    return self.PO[1,0,s] - self.PO_ic
                else:
                    return self.PO[1,0,s] - self.PO[1,0,1]
            else:
                return Expression.Skip
            
        self.PO_ice = Expression(self.s, rule=_init_PO)
        self.PO_icc = Constraint(self.s, rule=lambda self,s: self.PO_ice[s] == 0.0 if (1,s) in self.scenario_tree else Constraint.Skip)

        def _ode_PO_fed(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dPO_fed_dt[i,j,s] == self.F[i,s]*self.tf[i,s]*self.fe_dist[i]/self.PO_fed_scale
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.de_PO_fed = Constraint(self.fe_t, self.cp, self.s, rule=_ode_PO_fed)
          
        def _collocation_PO_fed(self,i,j,s): 
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dPO_fed_dt[i,j,s] == \
                           sum(self.ldot_t[j, k] * self.PO_fed[i,k,s] for k in self.cp)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.dvar_t_PO_fed = Constraint(self.fe_t, self.cp, self.s, rule=_collocation_PO_fed)
            
        def _continuity_PO_fed(self,i,s):
            if (i+1,s) in self.scenario_tree:
                if i < nfe and nfe > 1:
                    parent_s = self.scenario_tree[(i+1,s)][1] # parent scenario
                    return self.PO_fed[i+1,0,s] - sum(self.l1_t[j] * self.PO_fed[i,j,parent_s] for j in self.cp)
                else:
                    return Expression.Skip
            return Expression.Skip
        
        self.noisy_PO_fed = Expression(self.fe_t, self.s, rule=_continuity_PO_fed)
        self.cp_PO_fed = Constraint(self.fe_t, self.s, rule=lambda self,i,s: self.noisy_PO_fed[i,s] == 0.0 if (i+1,s) in self.scenario_tree and i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_PO_fed(self,s):
            if (1,s) in self.scenario_tree:
                if s == 1:
                    return self.PO_fed[1,0,s] - self.PO_fed_ic
                else:
                    return self.PO_fed[1,0,s] - self.PO_fed[1,0,1]
            else:
                return Expression.Skip
            
        self.PO_fed_ice = Expression(self.s, rule=_init_PO_fed)        
        self.PO_fed_icc = Constraint(self.s, rule=lambda self,s: self.PO_fed_ice[s] == 0.0 if (1,s) in self.scenario_tree else Constraint.Skip)

        def _ode_MX(self,i,j,o,s):
            if (i,s) in self.scenario_tree:
                if j > 0:   
                    if o == 0:
                        return self.dMX_dt[i,j,o,s] == (self.kr[i,j,'i',s]*self.G[i,j,s]*self.G_scale*self.PO[i,j,s]*self.PO_scale*self.Vi[i,j,s]*self.Vi_scale)/self.MX0_scale
                    elif o == 1:
                        return self.dMX_dt[i,j,o,s] == (self.kr[i,j,'i',s]*self.G[i,j,s]*self.G_scale+self.kr[i,j,'p',s]*self.MG[i,j,s])*self.PO[i,j,s]*self.PO_scale*self.Vi[i,j,s]*self.Vi_scale/self.MX1_scale
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.de_MX = Constraint(self.fe_t, self.cp, self.o, self.s, rule=_ode_MX)
        
        def _collocation_MX(self,i,j,o,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dMX_dt[i,j,o,s] == \
                           sum(self.ldot_t[j,k] * self.MX[i,k,o,s] for k in self.cp)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.dvar_t_MX = Constraint(self.fe_t, self.cp, self.o, self.s, rule=_collocation_MX)
        
        
        def _continuity_MX(self,i,o,s):
            if (i+1,s) in self.scenario_tree:
                if i < nfe and nfe > 1:
                    parent_s = self.scenario_tree[(i+1,s)][1] # parent scenario
                    return self.MX[i+1,0,o,s] - sum(self.l1_t[j] * self.MX[i,j,o,parent_s] for j in self.cp)
                else:
                    return Expression.Skip
            else:
                return Expression.Skip
            
        self.noisy_MX = Expression(self.fe_t, self.o, self.s, rule=_continuity_MX)
        self.cp_MX = Constraint(self.fe_t, self.o, self.s, rule=lambda self,i,o,s:self.noisy_MX[i,o,s] == 0.0 if (i+1,s) in self.scenario_tree and i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_MX(self,o,s):
            if (1,s) in self.scenario_tree:
                if s == 1:
                    return self.MX[1,0,o,s] - self.MX_ic[o]
                else:
                    return self.MX[1,0,o,s] - self.MX[1,0,o,1]
            else:
                return Expression.Skip
            
        self.MX_ice = Expression(self.o, self.s, rule=_init_MX)
        self.MX_icc = Constraint(self.o, self.s, rule=lambda self,o,s: self.MX_ice[o,s] == 0.0 if (1,s) in self.scenario_tree else Constraint.Skip)
    
        def _ode_Y(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dY_dt[i,j,s] == (self.kr[i,j,'t',s]*self.n_KOH*self.p_n_KOH[i,s]*self.PO[i,j,s]*self.PO_scale*self.Vi[i,j,s]*self.Vi_scale - self.kr[i,j,'i',s]*self.U[i,j,s]*self.U_scale*self.PO[i,j,s]*self.PO_scale*self.Vi[i,j,s]*self.Vi_scale)/self.Y_scale
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.de_Y = Constraint(self.fe_t, self.cp, self.s, rule=_ode_Y)
        
        
        def _collocation_Y(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dY_dt[i,j,s] == \
                           sum(self.ldot_t[j, k] * self.Y[i,k,s] for k in self.cp)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.dvar_t_Y = Constraint(self.fe_t, self.cp, self.s, rule=_collocation_Y)
        
        
        def _continuity_Y(self,i,s):
            if (i+1,s) in self.scenario_tree:
                if i < nfe and nfe > 1:
                    parent_s = self.scenario_tree[(i+1,s)][1] # parent scenario
                    return self.Y[i+1,0,s] - sum(self.l1_t[j] * self.Y[i,j,parent_s] for j in self.cp)
                else:
                    return Expression.Skip
            return Expression.Skip
        
        self.noisy_Y = Expression(self.fe_t, self.s, rule=_continuity_Y)
        self.cp_Y = Constraint(self.fe_t, self.s, rule=lambda self,i,s:self.noisy_Y[i,s] == 0.0 if (i+1,s) in self.scenario_tree and i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_Y(self,s):
            if (1,s) in self.scenario_tree:
                if s == 1:
                    return self.Y[1,0,s] - self.Y_ic
                else:
                    return self.Y[1,0,s] - self.Y[1,0,1]
            else:
                return Expression.Skip
            
        self.Y_ice = Expression(self.s, rule=_init_Y)
        self.Y_icc = Constraint(self.s, rule=lambda self,s: self.Y_ice[s] == 0.0 if (1,s) in self.scenario_tree else Constraint.Skip)
              
        def _ode_MY(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:   
                    return self.dMY_dt[i,j,s] == (self.kr[i,j,'i',s]*self.U[i,j,s]*self.U_scale*self.PO[i,j,s]*self.PO_scale*self.Vi[i,j,s]*self.Vi_scale)/self.MY0_scale
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.de_MY = Constraint(self.fe_t, self.cp, self.s, rule=_ode_MY)
        
        def _collocation_MY(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dMY_dt[i,j,s] == \
                           sum(self.ldot_t[j, k] * self.MY[i,k,s] for k in self.cp)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.dvar_t_MY = Constraint(self.fe_t, self.cp, self.s, rule=_collocation_MY)
        
        
        def _continuity_MY(self,i,s):
            if (i+1,s) in self.scenario_tree:
                if i < nfe and nfe > 1:
                    parent_s = self.scenario_tree[(i+1,s)][1] # parent scenario
                    return self.MY[i+1,0,s] - sum(self.l1_t[j] * self.MY[i,j,parent_s] for j in self.cp)
                else:
                    return Expression.Skip
            else:
                return Expression.Skip
            
        self.noisy_MY = Expression(self.fe_t, self.s, rule=_continuity_MY)
        self.cp_MY = Constraint(self.fe_t, self.s, rule=lambda self,i,s:self.noisy_MY[i,s] == 0.0 if (i+1,s) in self.scenario_tree and i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_MY(self,s):
            if (1,s) in self.scenario_tree:
                if s == 1:
                    return self.MY[1,0,s] - self.MY_ic
                else:
                    return self.MY[1,0,s] - self.MY[1,0,1]
            else:
                return Expression.Skip
        
        self.MY_ice = Expression(self.s, rule=_init_MY)
        self.MY_icc = Constraint(self.s, rule=lambda self,s: self.MY_ice[s] == 0.0 if (1,s) in self.scenario_tree else Constraint.Skip)
        
       
        # energy balance
        def _collocation_T_cw(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    # implicit systematic
                    #return self.dT_cw_dt[i] == sum(self.ldot_t[j, k] * self.T_cw[i, k]*self.T_scale for k in self.cp)/self.tf
                    # explicit tailored to piecewise-affine
                    return self.T_cw[i,j,s]*self.T_scale == self.T_cw[i,0,s]*self.T_scale + self.dT_cw_dt[i,s]*self.tf[i,s]*self.fe_dist[i]*self.tau_i_t[j]
                else:   
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.dvar_t_T_cw = Constraint(self.fe_t, self.cp, self.s, rule = _collocation_T_cw)
        
        def _continuity_T_cw(self,i,s):
            if (i+1,s) in self.scenario_tree:
                if i < nfe and nfe > 1:
                    parent_s = self.scenario_tree[(i+1,s)][1] # parent scenario
                    return self.T_cw[i+1,0,s] - sum(self.l1_t[j] * self.T_cw[i, j, parent_s] for j in self.cp)
                else:
                    return Expression.Skip 
            else:
                return Constraint.Skip

        self.noisy_T_cw = Expression(self.fe_t, self.s, rule=_continuity_T_cw)
        self.cp_T_cw = Constraint(self.fe_t, self.s, rule=lambda self,i,s:self.noisy_T_cw[i,s] == 0.0 if (i+1,s) in self.scenario_tree and i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_T_cw(self,s):
            if (1,s) in self.scenario_tree:
                if s == 1:
                    return self.T_cw[1,0,1] - self.T_cw_ic
                else:
                    return self.T_cw[1,0,s] - self.T_cw[1,0,1]
            else:
                return Constraint.Skip
            
        self.T_cw_ice = Expression(self.s, rule=_init_T_cw)
        self.T_cw_icc = Constraint(self.s, rule=lambda self,s: self.T_cw_ice[s] == 0.0 if (1,s) in self.scenario_tree else Constraint.Skip)
        
        def _ode_T(self,i,j,s):    
            if (i,s) in self.scenario_tree:
                if j > 0:
                    #return self.dT_dt[i,j]*(self.m_tot[i,j]*self.m_tot_scale)*(self.bulk_cp_1 + self.bulk_cp_2*self.T[i,j]*self.T_scale) == (-self.F[i]*self.Hrxn['p']*self.monomer_cooling[i,j]*self.tf*self.fe_dist[i] + \
                    #                          self.Hrxn['p']*(self.F[i]*self.tf*self.fe_dist[i] - self.dPO_dt[i,j]*self.PO_scale + self.dW_dt[i,j]*self.W_scale) - self.k_c*self.tf*self.fe_dist[i]*(self.T[i,j]*self.T_scale - self.T_cw[i]*self.T_scale))/self.T_scale
                    return self.dT_dt[i,j,s]*(self.m_tot[i,j,s]*self.m_tot_scale)*(self.bulk_cp_1 + self.bulk_cp_2*self.T[i,j,s]*self.T_scale) ==\
                                (self.Qr[i,j,s] - self.Qc[i,j,s] - self.F[i,s]*self.tf[i,s]*self.fe_dist[i]*self.mw_PO*self.monomer_cooling[i,j,s]*self.monomer_cooling_scale)*self.Hrxn['p']/self.T_scale
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
    
        self.de_T = Constraint(self.fe_t, self.cp, self.s, rule=_ode_T)         
        
        def _collocation_T(self, i, j, s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.dT_dt[i, j, s] == \
                           sum(self.ldot_t[j, k] * self.T[i, k, s] for k in self.cp)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.dvar_t_T = Constraint(self.fe_t, self.cp, self.s, rule=_collocation_T)
              
        def _continuity_T(self, i, s): 
            if (i+1,s) in self.scenario_tree:
                if i < nfe and nfe > 1:
                    parent_s = self.scenario_tree[(i+1,s)][1] # parent scenario
                    return self.T[i + 1, 0, s] - sum(self.l1_t[j] * self.T[i, j, parent_s] for j in self.cp)
                else:
                    return Expression.Skip
            else:
                return Expression.Skip
        
        self.noisy_T = Expression(self.fe_t, self.s, rule=_continuity_T)
        self.cp_T = Constraint(self.fe_t, self.s, rule=lambda self,i,s: self.noisy_T[i,s] == 0.0 if (i+1,s) in self.scenario_tree and i < nfe and nfe > 1 else Constraint.Skip)
        
        def _init_T(self,s):  
            if (1,s) in self.scenario_tree:
                if s == 1:
                    return self.T[1, 0, s] - self.T_ic
                else:
                    return self.T[1,0,s] - self.T[1,0,1]
            else:
                return Expression.Skip
        
        self.T_ice = Expression(self.s, rule=_init_T)
        self.T_icc = Constraint(self.s, rule=lambda self,s: self.T_ice[s] == 0.0 if (1,s) in self.scenario_tree else Constraint.Skip)      
        
        
        # kinetics
        def _rxn_rate_r_a(self,i,j,r,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return 0.0 == (self.T[i,j,s]*self.T_scale*log(self.p_A[r,i,s]*self.A[r]*self.scale*60*1000) - self.Ea[r]/self.Rg - self.T[i,j,s]*self.T_scale*self.k_l[i,j,r,s])
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.rxn_rate_r_a = Constraint(self.fe_t, self.cp, self.r, self.s, rule=_rxn_rate_r_a)

        def _rxn_rate_r_b(self,i,j,r,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return 0.0 == exp(self.k_l[i,j,r,s])*self.tf[i,s]*self.fe_dist[i] - self.kr[i,j,r,s]
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.rxn_rate_r_b = Constraint(self.fe_t, self.cp, self.r, self.s, rule=_rxn_rate_r_b)
        
        # algebraic equations
        def _ae_V(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j == 0:
                    return Constraint.Skip
                else:
                    return 0.0 == (1e3 - self.Vi[i,j,s]*self.Vi_scale * self.m_tot[i,j,s]*self.m_tot_scale*(1+0.0007576*(self.T[i,j,s]*self.T_scale - 298.15)))
            else:
                return Constraint.Skip
            
        self.ae_V = Constraint(self.fe_t, self.cp, self.s, rule=_ae_V)
        
        def _ae_equilibrium_a(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j == 0:
                    return Constraint.Skip
                else:
                    return 0.0 == self.G[i,j,s]*self.G_scale*(self.MX[i,j,0,s]*self.MX0_scale + self.MY[i,j,s]*self.MY0_scale + self.X[i,j,s]*self.X_scale + self.Y[i,j,s]*self.Y_scale) - self.X[i,j,s]*self.X_scale*self.n_KOH*self.p_n_KOH[i,s]
            else:
                return Constraint.Skip
            
        self.ae_equilibrium_a = Constraint(self.fe_t, self.cp, self.s, rule =_ae_equilibrium_a)
        
        def _ae_equilibrium_b(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j == 0:
                    return Constraint.Skip
                else:
                    return 0.0 == self.U[i,j,s]*self.U_scale*(self.MX[i,j,0,s]*self.MX0_scale + self.MY[i,j,s]*self.MY0_scale + self.X[i,j,s]*self.X_scale + self.Y[i,j,s]*self.Y_scale) - self.Y[i,j,s]*self.Y_scale*self.n_KOH*self.p_n_KOH[i,s]
            else:
                return Constraint.Skip
            
        self.ae_equilibrium_b = Constraint(self.fe_t, self.cp, self.s, rule =_ae_equilibrium_b)
        
        def _ae_equilibrium_c(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return 0.0 == self.MG[i,j,s]*(self.MX[i,j,0,s]*self.MX0_scale + self.MY[i,j,s]*self.MY0_scale + self.X[i,j,s]*self.X_scale + self.Y[i,j,s]*self.Y_scale) - self.MX[i,j,0,s]*self.MX0_scale*self.n_KOH*self.p_n_KOH[i,s]
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.ae_equilibrium_c = Constraint(self.fe_t, self.cp, self.s, rule =_ae_equilibrium_c)
        
        # constraints
        def _pc_heat_a(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j == 0:
                    return Constraint.Skip
                else:
                    return 0.0 == self.bulk_cp_1*(self.T[i,j,s]*self.T_scale) + self.bulk_cp_2*(self.T[i,j,s]*self.T_scale)**2/2.0 - self.int_T[i,j,s]*self.int_T_scale
            else:
                return Constraint.Skip
            
        self.pc_heat_a = Constraint(self.fe_t, self.cp, self.s, rule=_pc_heat_a)
        
        def _pc_heat_b(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j == 0:
                    return Constraint.Skip
                else:
                    return 0.0 == self.bulk_cp_1*self.Tad[i,j,s]*self.Tad_scale + self.bulk_cp_2*(self.Tad[i,j,s]*self.Tad_scale)**2/2.0 - self.int_Tad[i,j,s]*self.int_Tad_scale
            else:
                return Constraint.Skip
            
        self.pc_heat_b = Constraint(self.fe_t, self.cp, self.s, rule=_pc_heat_b)

        def _Q_in(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0: # normalized with delta_H_r
                    return self.Qr[i,j,s] == ((self.kr[i,j,'i',s]-self.kr[i,j,'p',s])*(self.G[i,j,s]*self.G_scale + self.U[i,j,s]*self.U_scale) + (self.kr[i,j,'p',s] + self.kr[i,j,'t',s])*self.n_KOH*self.p_n_KOH[i,s] + self.kr[i,j,'a',s]*self.W[i,j,s]*self.W_scale)*self.PO[i,j,s]*self.PO_scale*self.Vi[i,j,s]*self.Vi_scale*self.Hrxn_aux['p']*self.p_Hrxn_aux['p',i,s] + self.dW_dt[i,j,s]*self.W_scale * self.Hrxn_aux['p'] * self.p_Hrxn_aux['p',i,s]
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
        
        self.Q_in = Constraint(self.fe_t, self.cp, self.s, rule=_Q_in)
        
        def _Q_out(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return self.Qc[i,j,s] == self.kA * self.p_kA[i,s] * self.tf[i,s] * self.fe_dist[i] * (self.T[i,j,s]*self.T_scale - self.T_cw[i,j,s]*self.T_scale)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
        
        self.Q_out = Constraint(self.fe_t, self.cp, self.s, rule=_Q_out)
        
        def _pc_temp_a(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j == 0:
                    return Constraint.Skip
                else:
                    return 0.0 == self.m_tot[i,j,s]*self.m_tot_scale*(self.int_Tad[i,j,s]*self.int_Tad_scale - self.int_T[i,j,s]*self.int_T_scale) - self.PO[i,j,s]*self.PO_scale*self.Hrxn['p']*self.Hrxn_aux['p']*self.p_Hrxn_aux['p',i,s]
            else:
                return Constraint.Skip
            
        self.pc_temp_a = Constraint(self.fe_t, self.cp, self.s, rule=_pc_temp_a)
        
        def _pc_temp_b(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j == 0:
                    return Constraint.Skip
                else:
                    return 0.0 == (self.T_safety + self.Tb) - self.Tad[i,j,s]*self.Tad_scale - self.s_temp_b[i,j,s] + self.eps_pc[i,j,1,s]
            else:
                return Constraint.Skip
            
        self.pc_temp_b = Constraint(self.fe_t, self.cp, self.s, rule = _pc_temp_b)   
        
        def _pc_T_max(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return 0.0 == (self.T_max + self.Tb) - self.T[i,j,s]*self.T_scale - self.s_T_max[i,j,s] + self.eps_pc[i,j,2,s]
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.pc_T_max = Constraint(self.fe_t, self.cp, self.s, rule = _pc_T_max)
        
        def _pc_T_min(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j > 0:
                    return 0.0 == self.T[i,j,s]*self.T_scale - (self.T_min + self.Tb) - self.s_T_min[i,j,s] + self.eps_pc[i,j,3,s]
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
        
        self.pc_T_min = Constraint(self.fe_t, self.cp, self.s, rule = _pc_T_min)
        
        def _pc_heat_removal_c(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if j == 0:
                    return Constraint.Skip
                else:
                    return 0.0 == (self.mono_cp_1 * (self.T[i,j,s]*self.T_scale - self.feed_temp) + self.mono_cp_2/2.0 * ((self.T[i,j,s]*self.T_scale)**2.0 -self.feed_temp**2.0) + self.mono_cp_3/3.0*((self.T[i,j,s]*self.T_scale)**3.0 -self.feed_temp**3.0) + self.mono_cp_4/4.0*((self.T[i,j,s]*self.T_scale)**4.0-self.feed_temp**4.0)) - self.monomer_cooling[i,j,s]*self.monomer_cooling_scale*self.Hrxn['p']
            else:
                return Constraint.Skip
            
        self.pc_heat_removal_c = Constraint(self.fe_t, self.cp, self.s, rule=_pc_heat_removal_c)

        def _epc_PO_ptg(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if i == nfe and j == ncp:
                    return  0.0 == self.unreacted_PO*1e-6*self.m_tot[i,j,s]*self.m_tot_scale - self.PO[i,j,s]*self.PO_scale*self.mw_PO + self.eps[1,s] - self.s_PO_ptg[s]
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.epc_PO_ptg = Constraint(self.fe_t, self.cp, self.s, rule=_epc_PO_ptg)    
        
        def _epc_unsat(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if i == nfe and j == ncp:
                    return  0.0 == (self.unsat_value*self.m_tot[i,j,s]*self.m_tot_scale - 1000.0*(self.MY[i,j,s]*self.MY0_scale + self.Y[i,j,s]*self.Y_scale) + self.eps[2,s] - self.s_unsat[s]*100.0)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.epc_unsat = Constraint(self.fe_t, self.cp, self.s, rule=_epc_unsat)
        
        def _epc_PO_fed(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if i == nfe and j == ncp:
                    return 0.0 == self.PO_fed[i,j,s]*self.PO_fed_scale - self.n_PO - self.s_PO_fed[s]
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.epc_PO_fed = Constraint(self.fe_t, self.cp, self.s, rule=_epc_PO_fed)
        
        def _epc_mw(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if i == nfe and j == ncp:
                    return 0.0 == self.MX[i,j,1,s]*self.MX1_scale - (self.molecular_weight - self.mw_PG)/self.mw_PO/self.num_OH*self.MX[i,j,0,s]*self.MX0_scale + self.eps[3,s] - self.s_mw[s]
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.epc_mw = Constraint(self.fe_t, self.cp, self.s, rule=_epc_mw)
        
        def _epc_mw_ub(self,i,j,s):
            if (i,s) in self.scenario_tree:
                if i == nfe and j == ncp:
                    return 0.0 == self.MX[nfe,ncp,1,s]*self.MX1_scale - (self.molecular_weight_max + self.molecular_weight - self.mw_PG)/self.mw_PO/self.num_OH*self.MX[nfe,ncp,0,s]*self.MX0_scale - self.eps[4,s] + self.s_mw_ub[s]
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.epc_mw_ub = Constraint(self.fe_t, self.cp, self.s, rule=_epc_mw_ub)
        
        # controls (technicalities)
        self.u1_e = Expression(self.fe_t, self.s, rule = lambda self, i, s: self.dT_cw_dt[i,s] if (i,s) in self.scenario_tree else Expression.Skip)
        self.u2_e = Expression(self.fe_t, self.s, rule = lambda self, i, s: self.F[i,s] if (i,s) in self.scenario_tree else Expression.Skip)
        self.u1_c = Constraint(self.fe_t, self.s, rule = lambda self, i, s: self.u1_e[i,s] == self.u1[i,s] if (i,s) in self.scenario_tree else Constraint.Skip)
        self.u2_c = Constraint(self.fe_t, self.s, rule = lambda self, i, s: self.u2_e[i,s] == self.u2[i,s] if (i,s) in self.scenario_tree else Constraint.Skip)
        
        # non anticipativity
        def _non_anticipativity_u2(self,i,s):
            if (i,s) in self.scenario_tree:
                parent_node = self.scenario_tree[(i,s)][0:2]
                reference_node = self.scenario_tree[(i,s)][2]
                if reference_node:
                    return Constraint.Skip
                else:
                    for k in self.scenario_tree:
                        if self.scenario_tree[k][0:2] == parent_node and self.scenario_tree[k][2]:
                            aux = k
                            break
                    return 0.0 == self.u2[i,s] - self.u2[aux]
            else:
                return Constraint.Skip    
        
        self.non_anticipativity_u2 = Constraint(self.fe_t, self.s, rule=_non_anticipativity_u2)
        
        def _non_anticipativity_u1(self,i,s):
            if (i,s) in self.scenario_tree:
                parent_node = self.scenario_tree[(i,s)][0:2]
                reference_node = self.scenario_tree[(i,s)][2]
                if reference_node:
                    return Constraint.Skip
                else:
                    for k in self.scenario_tree:
                        if self.scenario_tree[k][0:2] == parent_node and self.scenario_tree[k][2]:
                            aux = k
                            break
                    return self.u1[i,s] - self.u1[aux] == 0.0
            else:
                return Constraint.Skip    
        
        self.non_anticipativity_u1 = Constraint(self.fe_t, self.s, rule=_non_anticipativity_u1)
        
        def _non_anticipativity_tf(self,i,s):
            if (i,s) in self.scenario_tree:
                parent_node = self.scenario_tree[(i,s)][0:2]
                reference_node = self.scenario_tree[(i,s)][2]
                if reference_node:
                    return Constraint.Skip
                else:
                    for k in self.scenario_tree:
                        if self.scenario_tree[k][0:2] == parent_node and self.scenario_tree[k][2]:
                            aux = k
                            break
                    return self.tf[i,s] - self.tf[aux] == 0.0
            else:
                return Constraint.Skip     
            #return Constraint.Skip
        
        self.non_anticipativity_tf = Constraint(self.fe_t, self.s, rule=_non_anticipativity_tf)
        
        # fix size of finite elements after robust horizon is reached 
        def _fix_element_size(self,i,s):
            if (i,s) in self.scenario_tree:
                parent_node = self.scenario_tree[(i,s)][0:2]
                if i > self.nr+1:
                    return 0 == self.tf[i,s] - self.tf[parent_node]
#                elif s == 1 and i != 1: #nominal scenario
#                    return 0 == self.tf[i,s] - self.tf[parent_node]
                else:
                    return Constraint.Skip
#                if (i,s) != (1,1):
#                    return 0 == self.tf[i,s] - self.tf[1,1]
#                else:
#                    return Constraint.Skip
            else:
                return Constraint.Skip
            
        self.fix_element_size = Constraint(self.fe_t, self.s, rule = _fix_element_size)
               
        self.s_tf = Var(self.fe_t, self.s, initialize=0.0, bounds=(0,None))
        
        def _robust_timestep(self,i,s):
            if (i,s) in self.scenario_tree and i > 1 and i < self.nr + 2:
                parent_node = self.scenario_tree[(i,s)][0:2]
                return 0.0 == self.tf[parent_node] - self.tf[i,s] - self.s_tf[i,s]
            else:
                return Constraint.Skip
            
        self.robust_timestep = Constraint(self.fe_t, self.s, rule=_robust_timestep)
        
        self.theta = Var(initialize=1000)
        self.s_theta = Var(self.s, initialize=0.0, bounds=(0.0,None))
        def _epi(self,s):
            if self.scenario_tree[(self.nfe,s)][2]:
                total_time = self.tf[self.nfe,s]
                for i in range(2,self.nfe+1):
                    parent_node = self.scenario_tree[(i,s)][0:2]
                    total_time += self.tf[parent_node]
                return 0.0 == self.theta - total_time - self.s_theta[s]
            else:
                return Constraint.Skip
                    
        self.epi = Constraint(self.s, rule=_epi)
        
        # objective
        # assumes symmetric tree, i.e. every node branches into the same number of children nodes
        # weights for obj. function:
        w = {}
        aux = 1.0
        s_max = sum(1 for key in self.scenario_tree if key[0] == self.nfe)
        for i in self.fe_t:
            aux *= sum(1 for key in self.scenario_tree if self.scenario_tree[key][:2] == self.scenario_tree[i,1][:2])
            w[i] = s_max/aux # total number of scenarios self.s_max  divided by sum over all scenarios
            
        def _eobj(self):
#            return 1.0/s_max*(sum(sum(self.tf[i,s]*w[i] for i in self.fe_t if (i,s) in self.scenario_tree) for s in self.s) \
#                    + self.rho*(sum(sum(self.eps[k,s] for s in self.s) for k in self.epc) \
#                    + sum(sum(sum(sum(self.eps_pc[i,j,k,s] for i in self.fe_t if (i,s) in self.scenario_tree) for s in self.s) for k in self.pc) for j in self.cp if j > 0)))
            return self.theta + self.rho*(sum(sum(self.eps[k,s] for s in self.s) for k in self.epc) + sum(sum(sum(sum(self.eps_pc[i,j,k,s]*w[i] for i in self.fe_t if (i,s) in self.scenario_tree) for s in self.s) for k in self.pc) for j in self.cp))
            
#            return 1.0/s_max*(sum(sum(self.tf[i,s]*w[i] for i in self.fe_t if (i,s) in self.scenario_tree) for s in self.s) \
#                    + self.rho*(sum(sum(self.eps[k,s] for s in self.s) for k in self.epc) \
#                    + sum(sum(sum(sum(self.eps_pc[i,j,k,s]*w[i] for i in self.fe_t if (i,s) in self.scenario_tree) for s in self.s) for k in self.pc) for j in self.cp if j > 0))\
#                    + self.gamma * sum((self.MX[self.nfe,self.ncp,1,s]*self.MX1_scale/(self.MX[self.nfe,self.ncp,0,s]*self.MX0_scale)*self.mw_PO*self.num_OH + self.mw_PG - self.molecular_weight)**2 for s in self.s))
#        self.epc_mw_ub.deactivate()
#        self.epc_mw.deactivate() 
        
        self.eobj = Objective(rule=_eobj,sense=minimize)
        
        #Suffixes
        self.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)                  
     
    def set_scenarios(self):
        for k in self.scenario_tree:
            #k[0] finite element
            #k[1] scenario
            try:
                for key in self.scenario_tree[1,1][3]:
                    p = getattr(self, 'p_' + key[0])
                    if type(key[1]) == tuple:
                        aux_key = key[1] + k
                    else:
                        aux_key = (key[1],k[0],k[1])
                    p[aux_key] = self.scenario_tree[k][3][key]
            except:
                # catch case if scenario_tree includes more scenarios than self.s does
                continue
            
    def e_state_relation(self):
        pass
    
    def multimodel(self):
        self.non_anticipativity_tf.deactivate()
        self.non_anticipativity_u1.deactivate()
        self.non_anticipativity_u2.deactivate()
        self.fix_element_size.deactivate()
        
        self.u1_multimodel = Constraint(self.fe_t, self.s, rule=lambda self,i,s: self.u1[i,s] == self.u1[i,1] if s != 1 else Constraint.Skip)
        self.u2_multimodel = Constraint(self.fe_t, self.s, rule=lambda self,i,s: self.u2[i,s] == self.u2[i,1] if s != 1 else Constraint.Skip)
        self.tf_multimodel = Constraint(self.fe_t, self.s, rule=lambda self,i,s: self.tf[i,s] == self.tf[1,1] if (i,s) in self.scenario_tree and (i,s) != (1,1) else Constraint.Skip)
        #self.tf_multimodel = Constraint(self.fe_t, self.s, rule=lambda self,i,s: self.tf[i,s] == self.tf[i,1] if s != 1 else Constraint.Skip)
        
        #clear bounds that are not required anymore
        dof = ['u1','u2','tf']
        for v in dof:
            var = getattr(self, v)
            for index in var.index_set():
                if index[-1] == 1:
                    continue
                else:
                    var[index].setlb(None)
                    var[index].setub(None)
        
        # use epigraph form if not only endtime is minimized
        if not(self.epc_mw_ub.active):
            for s in self.s:
                self.eps[3,s].fix(0.0)
                
            self.theta = Var(initialize=1e3, bounds=(0.0,None))
            self.epi = ConstraintList()        
            self.epc_mw.deactivate()
            for index in self.MW.index_set():
                i = index[0]
                j = index[1]
                s = 1
                if j != 0 and (i,s) in self.scenario_tree:
                    self.MW[index].value = (self.MX[i,j,1,s].value*self.MX1_scale/(self.MX[i,j,0,s].value*self.MX0_scale)*self.mw_PO.value*self.num_OH.value + self.mw_PG.value)/self.MW_scale
                
            for s in self.s:
                self.epi.add(sum(self.tf[i,s] for i in self.fe_t) \
                             + self.rho*(sum(self.eps[k,s] for k in self.epc) \
                             + sum(sum(sum(self.eps_pc[i,j,k,s] for i in self.fe_t)for k in self.pc) for j in self.cp if j > 0))\
                             #+ 10*(self.MW[self.nfe,self.ncp,s]*self.MW_scale-self.molecular_weight)**2.0
                             <= self.theta)
            self.eobj.expr = self.theta 
                
    def par_to_var(self):
        self.A['i'].setlb(396400.0*0.5/self.scale)
        self.A['i'].setub(396400.0*1.5/self.scale)
  
        self.A['p'].setlb(13504.2*0.5/self.scale)
        self.A['p'].setub(13504.2*1.5/self.scale)
        
        self.A['t'].setlb(1.509e6*0.5/self.scale)
        self.A['t'].setub(1.509e6*1.5/self.scale)
        
        self.Hrxn_aux['p'].setlb(0.5)
        self.Hrxn_aux['p'].setlb(1.5)
        
        self.kA.setlb(0.5*2200.0/self.Hrxn['p']*60/20.0)
        self.kA.setub(1.5*2200.0/self.Hrxn['p']*60/20.0)
    
    def create_output_relations(self):
        self.add_component('MW', Var(self.fe_t, self.cp, self.s, initialize=0.0, bounds=(0,None)))
        self.add_component('MW_c', Constraint(self.fe_t, self.cp, self.s))            
        self.MW_c.rule = lambda self,i,j,s: 0.0 == self.MX[i,j,1,s]*self.MX1_scale - (self.MW[i,j,s]*self.MW_scale - self.mw_PG)/self.mw_PO/self.num_OH*self.MX[i,j,0,s]*self.MX0_scale if j > 0 and (i,s) in self.scenario_tree else Constraint.Skip
        self.MW_c.reconstruct()
        
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
        self.pc_T_min.deactivate()
        self.pc_T_max.deactivate()
        self.robust_timestep.deactivate()
        
    def clear_aux_bounds(self):
        keep_bounds = ['s_theta','s_tf','s_temp_b','s_T_min','s_T_max','s_mw','s_PO_ptg','s_unsat','s_mw','s_mw_ub','s_PO_fed','eps','eps_pc','u1','u2','tf','T_cw','k_l'] 
        for var in self.component_objects(Var, active=True):
            if var.name in keep_bounds:
                continue
            else:
                for key in var.index_set():
                    var[key].setlb(None)
                    var[key].setub(None)

        # real degrees of freedom
        # u1, u2
        # tf[i,s] if self.scenario_tree[i,s][2] == True and i > self.nr + 1
        # idea behind that: 
        # choose nominal case as deciding factor for batch time --> equidistant
        #                                                       --> after robust horizon keep equidistantly spaced
       
        # handle control bounds better to avoid redundant bounds
        dof = ['u1','u2','tf']
        for d in dof:
            var = getattr(self, d)
            for key in var.index_set():
                if key in self.scenario_tree and not(self.scenario_tree[key][2]):
                    var[key].setlb(None)
                    var[key].setub(None)
                # tf is regarded as degree of freedom
#                if var.name == 'tf' and ((key[1] == 1 and key[0] > 1) or (key[0] > self.nr + 1)):
#                    var[key].setlb(None)
#                    var[key].setub(None)
                # tf[nr+1,s] yields the degree of freedom in this optimization problem
#                if var.name == 'tf' and (key[1] != 1 or key[0] != 1):
#                    var[key].setlb(None)
#                    var[key].setub(None)
                # tf_i >= tf_i+1
                if var.name == 'tf' and key[0] > self.nr + 1:
                    var[key].setlb(None)
                    var[key].setub(None)
                    
    def clear_all_bounds(self):
        for var in self.component_objects(Var):
            for key in var.index_set():
                var[key].setlb(None)
                var[key].setub(None)
                      
    def create_bounds(self):
        for s in self.s:
            for i in self.fe_t:
                self.tf[i,s].setlb(min(10.0,10.0*24.0/self.nfe))
                self.tf[i,s].setub(min(25.0,25.0*24.0/self.nfe))
                self.dT_cw_dt[i,s].setlb(-6.0)
                self.u1[i,s].setlb(-5.0)
                self.dT_cw_dt[i,s].setub(6.0)
                self.u1[i,s].setub(5.0)
                self.F[i,s].setlb(-1.0)
                self.u2[i,s].setlb(0.0)
                self.F[i,s].setub(4.0) 
                self.u2[i,s].setub(3.0)
                for j in self.cp:
                    self.T_cw[i,j,s].setlb(298.15/self.T_scale)
                    self.T_cw[i,j,s].setub((self.T_max + self.Tb)/self.T_scale)
                    self.T[i,j,s].setlb((25 + self.Tb)/self.T_scale)
                    self.T[i,j,s].setub((225 + self.Tb)/self.T_scale)
                    self.int_T[i,j,s].setlb((1.1*(100+self.Tb) + 2.72*(100+self.Tb)**2/2000)/self.int_T_scale)
                    self.int_T[i,j,s].setub((1.1*(170+self.Tb) + 2.72*(170+self.Tb)**2/2000)/self.int_T_scale)
                    self.Vi[i,j,s].setlb(0.9/self.Vi_scale*(1e3)/((self.m_KOH + self.m_PG + self.m_PO + self.m_H2O)*(1 + 0.0007576*((170+self.Tb)-298.15))))
                    self.Vi[i,j,s].setub(1.1/self.Vi_scale*(1e3)/((self.m_KOH + self.m_PG + self.m_H2O)*(1 + 0.0007576*((100+self.Tb)-298.15))))
                    self.Tad[i,j,s].setlb((100 + self.Tb)/self.Tad_scale)
                    self.Tad[i,j,s].setub((self.T_safety + self.Tb)/self.Tad_scale)        
                    for r in self.r:
                        self.k_l[i,j,r,s].setlb(None)
                        self.k_l[i,j,r,s].setub(20.0)

    def del_pc_bounds(self):
        for i in self.fe_t:
            for s in self.s:
                for j in self.cp:
                    self.int_T[i,j,s].setlb(None)
                    self.int_T[i,j,s].setub(None)
                    self.Tad[i,j,s].setlb(None)
                    self.Tad[i,j,s].setub(None)
                    self.Vi[i,j,s].setlb(0)
                    self.Vi[i,j,s].setub(None)
                    

    def write_nl(self):
        """Writes the nl file and the respective row & col"""
        name = str(self.__class__.__name__) + ".nl"
        self.write(filename=name,
                   format=ProblemFormat.nl,
                   io_options={"symbolic_solver_labels": True})
        
        
    def initialize_element_by_element(self):
        print('initializing element by element ...')
        m_aux = SemiBatchPolymerization_multistage(1,self.ncp)
        m_aux.deactivate_pc()
        m_aux.deactivate_epc()
        m_aux.eobj.deactivate()
        m_aux.F[1,1] = 1.0
        m_aux.dvar_t_T_cw.deactivate()
        m_aux.T_cw_icc.deactivate()
        m_aux.T_cw.fix(397.0/self.T_scale)
        m_aux.tf[1,1] = min(15.0*24.0/self.nfe,15.0)
        m_aux.F[1,1].fixed = True
        m_aux.tf[1,1].fixed = True
        opt = SolverFactory('ipopt')
        opt.options["halt_on_ampl_error"] = "yes"
        opt.options["max_iter"] = 5000
        opt.options["tol"] = 1e-5
        results = {}
        k = 0
        # solve square problem
        for fe_t in self.fe_t:
            results[fe_t] = m_aux.save_results(opt.solve(m_aux, tee=False, keepfiles=False))
            prevsol = results[fe_t]
            try:
                m_aux.T_ic = prevsol['T',(1,3,1)]
                m_aux.W_ic = prevsol['W',(1,3,1)]
                m_aux.PO_ic = prevsol['PO',(1,3,1)]
                m_aux.MY_ic = prevsol['MY',(1,3,1)]
                m_aux.Y_ic = prevsol['Y',(1,3,1)]
                m_aux.PO_fed_ic = prevsol['PO_fed',(1,3,1)]
                # initial guess for next element
                for i in m_aux.fe_t:
                    for j in m_aux.cp:
                        m_aux.T[i,j,1] = prevsol['T',(1,3,1)]
                        m_aux.MY[i,j,1] = prevsol['MY',(1,3,1)]
                        m_aux.W[i,j,1] = prevsol['W',(1,3,1)]
                        m_aux.PO[i,j,1] = prevsol['PO',(1,3,1)]
                        m_aux.m_tot[i,j,1] = prevsol['m_tot',(1,3,1)]
                        m_aux.X[i,j,1] = prevsol['X',(1,3,1)]
                        m_aux.Y[i,j,1] = prevsol['Y',(1,3,1)]
                        m_aux.PO_fed[i,j,1] = prevsol['PO_fed',(1,3,1)]
                # initial values for next element
                for o in m_aux.o:
                    m_aux.MX_ic[o] = prevsol['MX',(1,3,o,1)]
                    for i in m_aux.fe_t:
                        for j in m_aux.cp:
                            m_aux.MX[i,j,o,1] = prevsol['MX',(1,3,o,1)]
            except KeyError:
                print('     something went wrong during shifting element')
                m_aux.troubleshooting()
                #m_aux.pprint()
                break
            if results[fe_t]['solstat'] == ['ok','optimal']:
                print('----> element %i converged' % fe_t)
            else:
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
                        aux_key[len(aux_key)-1] = 1 # intialize every scenario by the same point
                        aux_key = tuple(aux_key)
                        var[key] = results[fe_t][var.name,aux_key]
                    else: # multiple indices
                        aux_key = 1
                        try:
                            var[key] = results[i+1][var.name,aux_key]
                        except KeyError:
                            pass
                    i+=1
   
            print('...initialization complete!')
        else:
            print('...initialization failed!')
            m_aux.p_A.pprint()
            m_aux.troubleshooting()
                
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
        
    def plot_profiles(self, var_list, control_list):
        #v = list of variables for which the profiles are plotted
        #only works for properties with 1 or two indices so far 
        k = 0
        for var in var_list:
            k += 1
            for s in range(1, self.s_max + 1):
                t = [] # time 
                x = [] # state variable
                t.append(0)#self.tf[key].value)
                key = (self.nfe,s)
                v = getattr(self,var)
                t_tot = 0.0
                for i in self.fe_t:
                    aux_key = (key[0],3,key[1])
                    x.append(v[aux_key].value)
                    t.append(t[-1]+self.tf[key].value)
                    t_tot += self.tf[key].value
                    key = self.scenario_tree[key][0:2]
                x.append(getattr(self,var+'_ic').value)
                x = np.array(x)
                t = np.array(t) 
                t = (-1)*t + t_tot 
                #print(t)
                plt.figure(k)
                plt.plot(t,x)
                plt.title(v.name+ '-profile')
                plt.ylabel(v.name)
                plt.xlabel('$t \, [min]$')  
             
        for control in control_list:
            k += 1
            for s in range(1, self.s_max + 1):
                t = [] # time 
                u = [] # state variable
                t.append(0)#self.tf[key].value)
                key = (self.nfe,s)
                v = getattr(self,control)
                t_tot = 0.0
                for i in self.fe_t:
                    aux_key = (key[0],key[1])
                    u.extend([v[aux_key].value]*2)
                    t.extend([t[-1]+self.tf[key].value]*2)
                    t_tot += self.tf[key].value
                    key = self.scenario_tree[key][0:2]
                del t[-1]
                u = np.array(u)
                t = np.array(t) 
                t = (-1)*t + t_tot 
                #print(t)
                plt.figure(k)
                plt.plot(t,u)
                plt.title(v.name+ '-profile')
                plt.ylabel(v.name)
                plt.xlabel('$t \, [min]$')  

    def check_feasibility(self, display = False):
        # evaluates the rhs for every endpoint constraint
        epsilon = {}
        i = self.nfe
        j = self.ncp  # assmuning radau nodes
        s = 1
        epsilon['epc_PO_ptg'] = self.unreacted_PO.value - self.PO[i,j,s].value*self.PO_scale*self.mw_PO.value/(self.m_tot[i,j,s].value*self.m_tot_scale)*1e6# 
        epsilon['epc_mw'] = self.MX[i,j,1,s].value*self.MX1_scale/(self.MX[i,j,0,s].value*self.MX0_scale)*self.mw_PO.value*self.num_OH.value + self.mw_PG.value - self.molecular_weight.value 
        epsilon['epc_unsat'] = self.unsat_value.value - 1000.0*(self.MY[i,j,s].value*self.MY0_scale + self.Y[i,j,s].value*self.Y_scale)/(self.m_tot[i,j,s].value*self.m_tot_scale) 
        
        if display:
            print(epsilon)
        return epsilon
    
    def get_NAMW(self):
        i = self.nfe
        j = self.ncp
        s = 1
        return self.mw_PG.value + self.mw_PO.value*self.num_OH.value*self.MX[i,j,1,s].value/self.MX[i,j,0,s].value
    
    def troubleshooting(self):
        with open("display.txt", "w") as f:
            self.display(ostream=f)
            f.close()
    
        with open("pprint.txt","w") as f:
            self.pprint(ostream=f)
            f.close()

## create scenario_tree
#s_max = 9
#nr = 1
#nfe = 24
#alpha = 0.2
#st = {}
#for i in range(1,nfe+1):
#    if i < nr + 1:
#        for s in range(1,s_max**i+1):
#            if s%s_max == 1:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),True,{('A',('p',)):1.0,('A',('i',)):1.0,('kA',()):1.0}) 
#            elif s%s_max == 2:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
#            elif s%s_max == 3:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
#            elif s%s_max == 4:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
#            elif s%s_max == 5:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
#            elif s%s_max == 6:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
#            elif s%s_max == 7:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
#            elif s%s_max == 8:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
#            else:
#                st[(i,s)] = (i-1,int(ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
#    else:
#        for s in range(1,s_max**nr+1):
#            st[(i,s)] = (i-1,s,True,st[(i-1,s)][3])
#
#            
#Solver = SolverFactory('ipopt')
#Solver.options["halt_on_ampl_error"] = "yes"
#Solver.options["max_iter"] = 1000
#Solver.options["tol"] = 1e-8
#Solver.options["linear_solver"] = "ma57"
#f = open("ipopt.opt", "w")
#f.write("print_info_string yes")
#f.close()
##
#m = SemiBatchPolymerization_multistage(nfe,3,robust_horizon=nr,s_max=s_max**nr,scenario_tree=st)
###
#m.initialize_element_by_element()
#m.create_output_relations()
#m.create_bounds()
#m.clear_aux_bounds()
#Solver.solve(m,tee=True)
#m.Tad_ic = Var(initialize=3)
#m.plot_profiles(var_list=['W', 'X', 'm_tot','MY','PO', 'T_cw', 'T', 'Tad'],control_list=['F'])
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
import numpy as np
from pyomo.dae import *
from pyomo.opt import ProblemFormat
from aux.cpoinsc import collptsgen
from aux.lagrange_f import lgr, lgry, lgrdot, lgrydot
import collections
import matplotlib.pyplot as plt
from six import itervalues, iterkeys, iteritems
import sys
import pickle
from scipy.integrate import *


# Scaling
W_scale = 1.0
Y_scale = 1.0e-2
PO_scale = 1.0e1
MY0_scale = 1.0e-1
MX0_scale = 1.0e1
MX1_scale = 1.0e2
MW_scale = 1.0e2
X_scale = 1.0
m_tot_scale = 1.0e4
T_scale = 1.0e2
Tad_scale = 1.0e2
Vi_scale = 1.0e-2
PO_fed_scale = 1.0e2
int_T_scale = 1.0e2
int_Tad_scale = 1.0e2
G_scale = 1.0
U_scale = 1.0e-2
monomer_cooling_scale = 1.0e-2
scale = 1.0

# 
n_KOH = 2.70005346641 # [kmol]
mw_PO = 58.08 # [kg/kmol]
bulk_cp_1 = 1.1 # [kJ/kg/K]
bulk_cp_2 = 2.72e-3
mono_cp_1 = 0.92 # [kJ/kg/K]
mono_cp_2 = 8.87e-3
mono_cp_3 = -3.10e-5
mono_cp_4 = 4.78e-8
Hrxn = 92048.0 # [kJ/kmol]
Hrxn_aux = 1.0
A_a = 8.64e4*60.0*1000.0 # [m^3/min/kmol]
A_i = 3.964e5*60.0*1000.0
A_p = 1.35042e4*60.0*1000.0
A_t = 1.509e6*60.0*1000.0
Ea_a = 82.425 # [kJ/kmol]
Ea_i = 77.822
Ea_p = 69.172
Ea_t = 105.018
kA = 2200.0*60.0/20.0/Hrxn # [kJ/min/K]
Tf = 298.15
nx = 10

# ODE System
def fun_gen(u1,u2):
    def fun(t,x):
        dxdt = [0.0]*nx
        PO = x[0]*PO_scale
        W = x[1]*W_scale
        Y = x[2]*Y_scale
        m_tot = x[3]*m_tot_scale
        MX0 = x[4]*MX0_scale
        MY = x[6]*MY0_scale
        T = x[7]*T_scale
        T_cw = x[8]*T_scale  
        if nx == 11:    
            F = x[10] #pwa
        else:
            F = u2 #pwc
        kr_i = A_i * np.exp(-Ea_i/(8.314e-3*T)) # [m^3/min/kmol] * exp([kJ/kmol]/[kJ/kmol/K]/[K]) = [m^3/min/kmol]
        kr_p = A_p * np.exp(-Ea_p/(8.314e-3*T))
        kr_a = A_a * np.exp(-Ea_a/(8.314e-3*T))
        kr_t = A_t * np.exp(-Ea_t/(8.314e-3*T)) 
        X = 10.0432852386*2 + 47.7348816877 - 2*W - MX0 # [kmol]
        G = X*n_KOH/(X+MX0+Y+MY) # [kmol]
        U = Y*n_KOH/(X+MX0+Y+MY) # [kmol]
        MG = MX0*n_KOH/(X+MX0+Y+MY) # [kmol]
        
        monomer_cooling = (mono_cp_1*(T - Tf) + mono_cp_2/2.0 *(T**2.0-Tf**2.0) + mono_cp_3/3.0*(T**3.0-Tf**3.0) + mono_cp_4/4.0*(T**4.0-Tf**4.0))/Hrxn #
        Vi = 1.0e3/m_tot/(1.0+0.0007576*(T - 298.15)) # 1/[m^3]
        Qr = (((kr_i-kr_p)*(G+U) + (kr_p + kr_t)*n_KOH + kr_a*W)*PO*Vi*Hrxn_aux - kr_a * W * PO * Vi * Hrxn_aux) # [m^3/min/kmol]*[kmol] *[kmol] *[1/m^3] * [kJ/kmol] = [kJ/min]
        Qc = kA * (T - T_cw) # [kJ/min/K] *[K] = [kJ/min]

        # PO
        dxdt[0] = (F - ((kr_i-kr_p)*(G+U)+(kr_p+kr_t)*n_KOH+kr_a*W) * PO * Vi)/PO_scale # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
        # W
        dxdt[1] = -kr_a * W * PO * Vi/W_scale # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
        # Y
        dxdt[2] = (kr_t*n_KOH-kr_i*U)*PO*Vi/Y_scale # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
        # m_tot
        dxdt[3] = mw_PO*F/m_tot_scale # [kg/kmol] * [kmol/min] = [kg/min]
        # MX0
        dxdt[4] = kr_i*G*PO*Vi/MX0_scale # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
        # MX1
        dxdt[5] = (kr_i*G+kr_p*MG)*PO*Vi/MX1_scale # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
        # MY  
        dxdt[6] = kr_i*U*PO*Vi/MY0_scale
        # T
        dxdt[7] = (Qr - Qc - F*monomer_cooling*mw_PO)*Hrxn/(m_tot*(bulk_cp_1 + bulk_cp_2*T))/T_scale # [kJ/min] / [kg] / [kJ/kg/K] = [K/min]
        # T_cw
        dxdt[8] = u1/T_scale
        # t
        dxdt[9] = 1.0
        # pwa F
        if nx == 11:
            dxdt[10] = u2
        else:
            pass
        return dxdt
    return fun

#x0 = [0.0]*9
#x0[0] = 0.0
#x0[1] = 10.043285238623751
#x0[2] = 0.0
#x0[3] = 0.138436*1e4
#x0[4] = 0.0
#x0[5] = 0.0
#x0[6] = 0.0
#x0[7] = 403.15
#x0[8] = 343.15
#x0[9] = 0.0

f = open('optimal_trajectory.pckl','rb')
traj = pickle.load(f)
if nx == 11:
    states = {'PO':[()],'MX':[(0,),(1,)],'MY':[()],'T':[()],'T_cw':[()],'Y':[()],'W':[()],'m_tot':[()],'F':[()]}
    stv = {('PO',()):0,('W',()):1,('Y',()):2,('m_tot',()):3,('MX',(0,)):4,('MX',(1,)):5,('MY',()):6,('T',()):7,('T_cw',()):8,('F',()):10}
    scaling = {('PO',()):1,('W',()):1.0,('Y',()):1,('m_tot',()):1,('MX',(0,)):1,('MX',(1,)):1,('MY',()):1,('T',()):1,('T_cw',()):1,('F',()):1}
else:
    states = {'PO':[()],'MX':[(0,),(1,)],'MY':[()],'T':[()],'T_cw':[()],'Y':[()],'W':[()],'m_tot':[()]}#,'F':[()]}
    stv = {('PO',()):0,('W',()):1,('Y',()):2,('m_tot',()):3,('MX',(0,)):4,('MX',(1,)):5,('MY',()):6,('T',()):7,('T_cw',()):8}#,('F',()):10}
    scaling = {('PO',()):1,('W',()):1.0,('Y',()):1,('m_tot',()):1,('MX',(0,)):1,('MX',(1,)):1,('MY',()):1,('T',()):1,('T_cw',()):1}#,('F',()):1}
# set initial guess
x0 = [0.0]*nx
for x in states:
    for key in states[x]:
        index = stv[(x,key)]
        x0[index] = traj[x,(1,0)+key]*scaling[(x,key)]
x0[9] = 0.0
print(x0)
#u1 = traj['u1',1]
#u2 = traj['u2',1]
#delta_t = traj['tf',None]
#fx = fun_gen(u1,u2)
#out = RK45(fx,0.0,x0,delta_t)

delta_t = traj['tf',None]
nfe = 24
results = {}
timesteps = 3
u1_l = []
u2_l = []
for i in range(1,nfe+1):
    u1 = traj['u1',i]
    u1_l.append(u1)
    u2 = traj['u2',i]
    u2_l.append(u2)
    fx = fun_gen(u1,u2)
    #sol = odeint(fx,x0,t)
    #t = np.linspace(0.0,delta_t,timesteps)
    #t = [0.0*delta_t, 0.15505102572168217*delta_t, 0.64494897427831777*delta_t, 1.0*delta_t]
    sol = solve_ivp(fx,(0.0,delta_t),x0,method='Radau')#t_eval=t)
    results[i] = sol.y
    x0 = [results[i][k][-1] for k in range(len(x0))]


# plot results
l = 0
t = sum([[results[k][9][i] for i in range(len(results[k][9][:]))] for k in range(1,nfe+1)],[])
#t = [results[k][-1][9] for k in range(1,nfe+1)]
for xvar in states:
    for key in states[xvar]:
        plt.figure(l)
        index = stv[(xvar,key)]
        x = sum([[results[k][index][i] for i in range(len(results[k][9][:]))] for k in range(1,nfe+1)],[])
        #x = [results[k][-1][index] for k in range(1,nfe+1)]
        if key == ():
            name = xvar
        else:
            name = xvar + str(key)
        plt.plot(t,x,label=name)
        plt.legend()
        l += 1
    

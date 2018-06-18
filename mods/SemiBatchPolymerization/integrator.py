from scipy.integrate import *
from six import itervalues, iterkeys, iteritems
import matplotlib.pyplot as plt
import numpy as np
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
kA = 2200.0*60.0/10.0/Hrxn # [kJ/min/K]
Tf = 298.15
nx = 10
n_p = 3
p = [1.0]*n_p

#def fun_gen(D,F):
#    def fun(t,x):
#        dxdt = [0.0]*nx
#        PO = x[0]*PO_scale
#        W = x[1]*W_scale
#        Y = x[2]*Y_scale
#        m_tot = x[3]*m_tot_scale
#        MX0 = x[4]*MX0_scale
#        MY = x[6]*MY0_scale
#        T = x[7]*T_scale
#        T_cw = x[8]*T_scale  
#        kr_i = A_i * np.exp(-Ea_i/(8.314e-3*T)) # [m^3/min/kmol] * np.exp([kJ/kmol]/[kJ/kmol/K]/[K]) = [m^3/min/kmol]
#        kr_p = A_p * np.exp(-Ea_p/(8.314e-3*T))
#        kr_a = A_a * np.exp(-Ea_a/(8.314e-3*T))
#        kr_t = A_t * np.exp(-Ea_t/(8.314e-3*T)) 
#        X = 10.0432852386*2 + 47.7348816877 - 2*W - MX0 # [kmol]
#        G = X*n_KOH/(X+MX0+Y+MY) # [kmol]
#        U = Y*n_KOH/(X+MX0+Y+MY) # [kmol]
#        MG = MX0*n_KOH/(X+MX0+Y+MY) # [kmol]
#        
#        monomer_cooling = (mono_cp_1*(T - Tf) + mono_cp_2/2.0 *(T**2.0-Tf**2.0) + mono_cp_3/3.0*(T**3.0-Tf**3.0) + mono_cp_4/4.0*(T**4.0-Tf**4.0))/Hrxn #
#        Vi = 1.0e3/m_tot/(1.0+0.0007576*(T - 298.15)) # 1/[m^3]
#        Qr = (((kr_i-kr_p)*(G+U) + (kr_p + kr_t)*n_KOH + kr_a*W)*PO*Vi*Hrxn_aux - kr_a * W * PO * Vi * Hrxn_aux)*12.62 # [m^3/min/kmol]*[kmol] *[kmol] *[1/m^3] * [kJ/kmol] = [kJ/min]
#        Qc = kA * (T - T_cw)*12.62 # [kJ/min/K] *[K] = [kJ/min]
#
#        # PO
#        dxdt[0] = (F - ((kr_i-kr_p)*(G+U)+(kr_p+kr_t)*n_KOH+kr_a*W) * PO * Vi)/PO_scale*12.62 # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
#        # W
#        dxdt[1] = -kr_a * W * PO * Vi/W_scale*12.62 # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
#        # Y
#        dxdt[2] = (kr_t*n_KOH-kr_i*U)*PO*Vi/Y_scale*12.62 # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
#        # m_tot
#        dxdt[3] = mw_PO*F/m_tot_scale*12.62 # [kg/kmol] * [kmol/min] = [kg/min]
#        # MX0
#        dxdt[4] = kr_i*G*PO*Vi/MX0_scale*12.62 # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
#        # MX1
#        dxdt[5] = (kr_i*G+kr_p*MG)*PO*Vi/MX1_scale*12.62 # [m^3/kmol/min] * [kmol] * [kmol] * [1/m^3] = [kmol/min]
#        # MY  
#        dxdt[6] = kr_i*U*PO*Vi/MY0_scale*12.62
#        # T
#        dxdt[7] = (Qr - Qc - F*monomer_cooling*mw_PO*12.62)*Hrxn/(m_tot*(bulk_cp_1 + bulk_cp_2*T))/T_scale # [kJ/min] / [kg] / [kJ/kg/K] = [K/min]
#        # T_cw
#        dxdt[8] = (D*(353.15-T_cw)*12.62 + Qc*Hrxn/5e3/4.18)/T_scale
#        # t
#        dxdt[9] = 1.0
#        print(Qc,Qr,monomer_cooling)

        
def fun_gen(D,F):
    def fun(t,x):  
        dxdt = [0.0]*nx
#        dxdt[0] = (F - ((A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))-A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale)))*((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)+x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale))+(A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale))+A_t * np.exp(-Ea_t/(8.314e-3*x[7]*T_scale)))*n_KOH+A_a * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale))*x[1]*W_scale) * x[0]*PO_scale * 1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15)))/PO_scale;
#        dxdt[1] = -A_a * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale)) * x[1]*W_scale * x[0]*PO_scale * 1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/W_scale;
#        dxdt[2] = (A_t * np.exp(-Ea_t/(8.314e-3*x[7]*T_scale))*n_KOH-A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale))*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/Y_scale; 
#        dxdt[3] = mw_PO*F/m_tot_scale;
#        dxdt[4] = A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*(10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/MX0_scale;
#        dxdt[5] = (A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*(10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)+A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale))*x[4]*MX0_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale))*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/MX1_scale;
#        dxdt[6] = A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/MY0_scale;
#        dxdt[7] = ((((A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))-A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale)))*((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)+x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)) + (A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale)) + A_t * np.exp(-Ea_t/(8.314e-3*x[7]*T_scale)))*n_KOH + A_a * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale))*x[1]*W_scale)*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))*Hrxn_aux - A_a * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale)) * x[1]*W_scale * x[0]*PO_scale * 1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15)) * Hrxn_aux) - kA * (x[7]*T_scale - x[8]*T_scale) - F*((mono_cp_1*((x[7]*T_scale) - Tf) + mono_cp_2/2.0 *(pow(x[7]*T_scale,2.0)-pow(Tf,2.0)) + mono_cp_3/3.0*(pow(x[7]*T_scale,3.0)-pow(Tf,3.0)) + mono_cp_4/4.0*(pow(x[7]*T_scale,4.0)-pow(Tf,4.0)))/Hrxn)*mw_PO)*Hrxn/(x[3]*m_tot_scale*(bulk_cp_1 + bulk_cp_2*x[7]*T_scale))/T_scale;
#        dxdt[8] = (D*(353.15-x[8]*T_scale) + kA * p[2] * (x[7]*T_scale - x[8]*T_scale) * Hrxn/5e3/4.18)/T_scale;
#        dxdt[9] = 1.0
        Qr = (((A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))-A_p*p[0] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale)))*(x[2]*Y_scale+10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/(10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale) + (A_p*p[0] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale)) + A_t*p[0] * np.exp(-Ea_t/(8.314e-3*x[7]*T_scale)))*n_KOH + A_a*p[0] * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale))*x[1]*W_scale)*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0 + 0.0007576*(x[7]*T_scale - 298.15))*Hrxn_aux - A_a*p[0] * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale)) * x[1]*W_scale * x[0]*PO_scale *1.0e3/(x[3]*m_tot_scale)/(1.0 + 0.0007576*(x[7]*T_scale - 298.15))*Hrxn_aux)# [m^3/min/kmol]*[kmol] *[kmol] *[1/m^3] * [kJ/kmol] = [kJ/min]
        dxdt[0] = (F - ((A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))-A_p * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale)))*((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)+x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale))+(A_p * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale))+A_t * np.exp(-Ea_t/(8.314e-3*x[7]*T_scale)))*n_KOH+A_a * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale))*x[1]*W_scale) * x[0]*PO_scale * 1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15)))/PO_scale
        dxdt[1] = -A_a * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale)) * x[1]*W_scale * x[0]*PO_scale * 1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/W_scale
        dxdt[2] = (A_t * np.exp(-Ea_t/(8.314e-3*x[7]*T_scale))*n_KOH-A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale))*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/Y_scale
        dxdt[3] = mw_PO*F/m_tot_scale
        dxdt[4] = A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*(10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/MX0_scale
        dxdt[5] = (A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*(10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)+A_p * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale))*x[4]*MX0_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale))*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/MX1_scale
        dxdt[6] = A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/MY0_scale
        dxdt[7] = (Qr - kA * (x[7]*T_scale - x[8]*T_scale) - F*((mono_cp_1*((x[7]*T_scale) - Tf) + mono_cp_2/2.0 *(pow(x[7]*T_scale,2.0)-pow(Tf,2.0)) + mono_cp_3/3.0*(pow(x[7]*T_scale,3.0)-pow(Tf,3.0)) + mono_cp_4/4.0*(pow(x[7]*T_scale,4.0)-pow(Tf,4.0)))/Hrxn)*mw_PO)*Hrxn/(x[3]*m_tot_scale*(bulk_cp_1 + bulk_cp_2*x[7]*T_scale))/T_scale
        dxdt[8] = (D*(353.15-x[8]*T_scale) + kA * (x[7]*T_scale - x[8]*T_scale) * Hrxn/5e3/4.18)/T_scale
        dxdt[9] = 1.0
        return dxdt
    return fun
#x0 = [0.0,10.0432852386,0.0,0.138436,0.0,0.0,0.0,3.8315,3.8315,0.0]
#fx = fun_gen(0.35,3.0)
#sol = solve_ivp(fx,(0.0,12.62),x0,method='RK45')
#results = sol.y
#plt.figure(1)
#plt.plot(results[-1][:],results[0][:])
#plt.figure(2)
#plt.plot(results[-1][:],results[7][:])
#plt.figure(3)
#plt.plot(results[-1][:],results[2][:])
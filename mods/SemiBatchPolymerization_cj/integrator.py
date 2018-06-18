from scipy.integrate import *
from six import itervalues, iterkeys, iteritems
import matplotlib.pyplot as plt
import numpy as np
from mod_class_cj import *  
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
#n_p = 3
#p = [1.0]*n_p

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

        
def fun_gen(dTcwdt,F,p=[1,1,1]):
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
        Qr = (((A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))-A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale)))*(x[2]*Y_scale+10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/(10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale) + (A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale)) + A_t*p[0] * np.exp(-Ea_t/(8.314e-3*x[7]*T_scale)))*n_KOH + A_a*p[0] * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale))*x[1]*W_scale)*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0 + 0.0007576*(x[7]*T_scale - 298.15))*Hrxn_aux - A_a*p[0] * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale)) * x[1]*W_scale * x[0]*PO_scale *1.0e3/(x[3]*m_tot_scale)/(1.0 + 0.0007576*(x[7]*T_scale - 298.15))*Hrxn_aux)# [m^3/min/kmol]*[kmol] *[kmol] *[1/m^3] * [kJ/kmol] = [kJ/min]
        dxdt[0] = (F - ((A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))-A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale)))*((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)+x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale))+(A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale))+A_t * np.exp(-Ea_t/(8.314e-3*x[7]*T_scale)))*n_KOH+A_a * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale))*x[1]*W_scale) * x[0]*PO_scale * 1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15)))/PO_scale
        dxdt[1] = -A_a * np.exp(-Ea_a/(8.314e-3*x[7]*T_scale)) * x[1]*W_scale * x[0]*PO_scale * 1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/W_scale
        dxdt[2] = (A_t * np.exp(-Ea_t/(8.314e-3*x[7]*T_scale))*n_KOH-A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale))*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/Y_scale
        dxdt[3] = mw_PO*F/m_tot_scale
        dxdt[4] = A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*(10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/MX0_scale
        dxdt[5] = (A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*(10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)+A_p*p[1] * np.exp(-Ea_p/(8.314e-3*x[7]*T_scale))*x[4]*MX0_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale))*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/MX1_scale
        dxdt[6] = A_i*p[0] * np.exp(-Ea_i/(8.314e-3*x[7]*T_scale))*x[2]*Y_scale*n_KOH/((10.0432852386*2 + 47.7348816877 - 2*x[1]*W_scale - x[4]*MX0_scale)+x[4]*MX0_scale+x[2]*Y_scale+x[6]*MY0_scale)*x[0]*PO_scale*1.0e3/(x[3]*m_tot_scale)/(1.0+0.0007576*(x[7]*T_scale - 298.15))/MY0_scale
        dxdt[7] = (Qr - kA*p[2] * (x[7]*T_scale - x[8]*T_scale) - F*((mono_cp_1*((x[7]*T_scale) - Tf) + mono_cp_2/2.0 *(pow(x[7]*T_scale,2.0)-pow(Tf,2.0)) + mono_cp_3/3.0*(pow(x[7]*T_scale,3.0)-pow(Tf,3.0)) + mono_cp_4/4.0*(pow(x[7]*T_scale,4.0)-pow(Tf,4.0)))/Hrxn)*mw_PO)*Hrxn/(x[3]*m_tot_scale*(bulk_cp_1 + bulk_cp_2*x[7]*T_scale))/T_scale
        dxdt[8] = dTcwdt/T_scale#(D*(353.15-x[8]*T_scale) + kA * (x[7]*T_scale - x[8]*T_scale) * Hrxn/5e3/4.18)/T_scale
        dxdt[9] = 1.0
        return dxdt
    return fun

Solver = SolverFactory('ipopt')
Solver.options["halt_on_ampl_error"] = "yes"
Solver.options["max_iter"] = 5000
Solver.options["tol"] = 1e-8
Solver.options["linear_solver"] = "ma57"
f = open("ipopt.opt", "w")
f.write("print_info_string yes")
f.close()

e = SemiBatchPolymerization(24,3)
e.initialize_element_by_element()
e.create_bounds()
e.clear_aux_bounds()
e.tf.fix(24.2843333636)
u1 = [-0.19345919296144892,
 0.1859650040055502,
 -0.03494925359599763,
 0.03380507701931857,
 0.0006175233955904917,
 0.012806655027096787,
 0.0061003384620349835,
 0.0027466978476645546,
 0.0003299914682169199,
 0.00020436925270081522,
 0.0002712507805506018,
 0.00020179238506816063,
 9.077855132901096e-05,
 3.678647793951325e-05,
 -1.888981181001188e-05,
 -0.001533968287745645,
 -0.01023272431346022,
 0.01942375563134745,
 0.05264141935533248,
 0.011210085378191957,
 -0.20409581883221475,
 1.2859066636070562,
 0.8703084012819619,
 -0.028533529004159067]
u2 = [1.7665775585193184,
 1.401591518064247,
 1.1906138491066938,
 1.0709175164047342,
 1.018475914284693,
 0.9711223695200286,
 0.9514841988347906,
 0.9420194883708382,
 0.9374144893284764,
 0.93425152606075,
 0.9327592977752731,
 0.9323038852150022,
 0.9322303322860322,
 0.9322191149506205,
 0.9322200085794975,
 0.9323658449304979,
 0.933562017412809,
 0.9144389754680584,
 0.9086296031863798,
 0.9735481101147783,
 1.1098719051293335,
 0.0,
 0.0,
 0.0]
for i in range(1,25):
    e.u1[i].fix(u1[i-1])
    e.u2[i].fix(u2[i-1])
e.deactivate_pc()
e.deactivate_epc()
e.clear_all_bounds()
e.eobj.deactivate()
Solver.solve(e,tee=True)
u1 = [e.u1[i].value for i in e.u1.index_set()]
u2 = [e.u2[i].value for i in e.u1.index_set()]

tf = e.tf.value    
results = {}
grid = [-0.2,-0.1,0.0,0.1,0.2]
kA_ha = np.array(grid)#np.linspace(-0.2,0.2,num=4)
Ap = np.array(grid)#np.linspace(-0.2,0.2,num=4)
Ai = np.array(grid)#np.linspace(-0.2,0.2,num=4)
Ap, Ai, kA_ha = np.meshgrid(kA_ha, Ap, Ai)
i = 0
scenarios = {}
n = len(grid)
for j in range(n):
    for k in range(n):
        for l in range(n):
            scenarios[i] = {('A',('p',)):Ap[j][k][l],('A',('i',)):Ai[j][k][l],('kA',()):kA_ha[j][k][l]}
            i += 1
            
for k in range(n**3):
    x0 = [0.0,10.0432852386,0.0,0.138436,0.0,0.0,0.0,3.9315,3.7315,0.0]
    for i in range(1,25):
        fx = fun_gen(u1[i-1],u2[i-1],p=[1+scenarios[k][('A',('i',))],1+scenarios[k][('A',('p',))],1+scenarios[k][('kA',())]])
        sol = solve_ivp(fx,(0.0,tf),x0,method='BDF')
        results[k,i] = sol.y
        x0 = results[k,i][:,-1]

# comparison:
for k in range(n**3):
    t_int = [results[k,i][9,j] for i in range(1,25) for j in range(len(results[k,i][7,:]))]
    T_int = [results[k,i][7,j] for i in range(1,25) for j in range(len(results[k,i][7,:]))]
    plt.plot(t_int,T_int,'grey')
 
for k in range(n**3):
    Tad_int = []
    PO_int = []
    t_int = [results[k,i][9,j] for i in range(1,25) for j in range(len(results[k,i][7,:]))]
    for i in range(1,25):
        for j in range(len(results[k,i][7,:])):
            T = results[k,i][7,j]
            PO = results[k,i][0,j]
            m_tot = results[k,i][3,j]
            int_T = bulk_cp_1*(T*T_scale) + bulk_cp_2*(T*T_scale)**2/2.0 
            int_Tad = int_T + PO*PO_scale/(m_tot*m_tot_scale)*Hrxn
            Tad_int.append(-1/2.0*e.bulk_cp_1/e.bulk_cp_2*e.Tad_scale/(e.Tad_scale**2)*2.0 + np.sqrt((1/2.0*e.bulk_cp_1/e.bulk_cp_2*e.Tad_scale/(e.Tad_scale**2)*2.0)**2.0 + 2.0/e.bulk_cp_2*int_Tad/(e.Tad_scale**2)))
            PO_int.append(PO)
    plt.figure(2)
    plt.plot(t_int,Tad_int,'grey')
"""
t_int = [results[i][9,j] for i in range(1,25) for j in range(len(results[i][7,:]))]
T_int = [results[i][7,j] for i in range(1,25) for j in range(len(results[i][7,:]))]
#
t_radau = [0]
T = [e.T[1,0].value]
Tad = [e.T[1,0].value]
for i in range(1,25):
    for cp in range(1,4):
        t_radau.append(t_radau[-cp]+e.tau_i_t[cp]*tf)
        T.append(e.T[i,cp].value)
        Tad.append(e.Tad[i,cp].value)
Tad_int = []
PO_int = []
for i in range(1,25):
    for j in range(len(results[i][7,:])):
        T = results[i][7,j]
        PO = results[i][0,j]
        m_tot = results[i][3,j]
        int_T = bulk_cp_1*(T*T_scale) + bulk_cp_2*(T*T_scale)**2/2.0 
        int_Tad = int_T + PO*PO_scale/(m_tot*m_tot_scale)*Hrxn
        Tad_int.append(-1/2.0*e.bulk_cp_1/e.bulk_cp_2*e.Tad_scale/(e.Tad_scale**2)*2.0 + np.sqrt((1/2.0*e.bulk_cp_1/e.bulk_cp_2*e.Tad_scale/(e.Tad_scale**2)*2.0)**2.0 + 2.0/e.bulk_cp_2*int_Tad/(e.Tad_scale**2)))
        PO_int.append(PO)
plt.figure(2)
plt.plot(t_int,Tad_int)
#plt.plot(t_radau,Tad,color='red')
plt.plot([0,300],[4.4315,4.4315])

### lagrange interpolation polynomials

t_lagrange = [0]
T = []
Tad = []
Tad2 = []
m_tot = []
PO_l = []
lagrange_poly = {}
for i in range(1,25):
    for t in np.linspace(0,1,1000/20):
        lagrange_poly[0] = (t-e.tau_i_t[1])*(t-e.tau_i_t[2])*(t-e.tau_i_t[3])/(e.tau_i_t[0]-e.tau_i_t[1])/(e.tau_i_t[0]-e.tau_i_t[2])/(e.tau_i_t[0]-e.tau_i_t[3])
        lagrange_poly[1] = (t-e.tau_i_t[0])*(t-e.tau_i_t[2])*(t-e.tau_i_t[3])/(e.tau_i_t[1]-e.tau_i_t[0])/(e.tau_i_t[1]-e.tau_i_t[2])/(e.tau_i_t[1]-e.tau_i_t[3])
        lagrange_poly[2] = (t-e.tau_i_t[0])*(t-e.tau_i_t[1])*(t-e.tau_i_t[3])/(e.tau_i_t[2]-e.tau_i_t[0])/(e.tau_i_t[2]-e.tau_i_t[1])/(e.tau_i_t[2]-e.tau_i_t[3])
        lagrange_poly[3] = (t-e.tau_i_t[0])*(t-e.tau_i_t[1])*(t-e.tau_i_t[2])/(e.tau_i_t[3]-e.tau_i_t[0])/(e.tau_i_t[3]-e.tau_i_t[1])/(e.tau_i_t[3]-e.tau_i_t[2])
        T.append(lagrange_poly[0]*e.T[i,0].value+lagrange_poly[1]*e.T[i,1].value+lagrange_poly[2]*e.T[i,2].value+lagrange_poly[3]*e.T[i,3].value)
        T_now = T[-1]
        PO=lagrange_poly[0]*e.PO[i,0].value+lagrange_poly[1]*e.PO[i,1].value+lagrange_poly[2]*e.PO[i,2].value+lagrange_poly[3]*e.PO[i,3].value
        PO_l.append(lagrange_poly[0]*e.PO[i,0].value+lagrange_poly[1]*e.PO[i,1].value+lagrange_poly[2]*e.PO[i,2].value+lagrange_poly[3]*e.PO[i,3].value)
        m_tot=(lagrange_poly[0]*e.PO_fed[i,0].value+lagrange_poly[1]*e.PO_fed[i,1].value+lagrange_poly[2]*e.PO_fed[i,2].value+lagrange_poly[3]*e.PO_fed[i,3].value)*e.PO_fed_scale/e.m_tot_scale*e.mw_PO.value+e.m_tot_ic.value
        int_T = bulk_cp_1*(T_now*T_scale) + bulk_cp_2*(T_now*T_scale)**2/2.0 
        int_Tad = int_T + PO*PO_scale/(m_tot*m_tot_scale)*Hrxn
        Tad.append(-1/2.0*e.bulk_cp_1/e.bulk_cp_2*e.Tad_scale/(e.Tad_scale**2)*2.0 + np.sqrt((1/2.0*e.bulk_cp_1/e.bulk_cp_2*e.Tad_scale/(e.Tad_scale**2)*2.0)**2.0 + 2.0/e.bulk_cp_2*int_Tad/(e.Tad_scale**2)))
        #if i > 1:
        #    Tad2.append(lagrange_poly[0]*e.Tad[i-1,3].value+lagrange_poly[1]*e.Tad[i,1].value+lagrange_poly[2]*e.Tad[i,2].value+lagrange_poly[3]*e.Tad[i,3].value)
        #else:
        #    Tad2.append(lagrange_poly[0]*e.T[1,0].value+lagrange_poly[1]*e.Tad[i,1].value+lagrange_poly[2]*e.Tad[i,2].value+lagrange_poly[3]*e.Tad[i,3].value)
t_lagrange = np.linspace(0,24*tf,1000*24/20)
plt.plot(t_lagrange,Tad)
#plt.plot(t_lagrange,Tad2)

plt.figure(3)
plt.plot(t_int,T_int)
plt.plot(t_lagrange,T,'red')

# PO
plt.figure(4)
plt.plot(t_int,PO_int)
plt.plot(t_lagrange,PO_l,'red')
"""
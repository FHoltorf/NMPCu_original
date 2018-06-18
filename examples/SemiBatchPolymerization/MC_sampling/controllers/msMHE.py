from __future__ import print_function
from main.mods.SemiBatchPolymerization.mod_class_multistage import SemiBatchPolymerization_multistage
from main.mods.SemiBatchPolymerization.mod_class import SemiBatchPolymerization
from main.dync.MHEGen import msMHEGen
from main.examples.SemiBatchPolymerization.noise_characteristics import * 
import numpy as np

# don't write messy bytecode files
# might want to change if ran multiple times for performance increase
# sys.dont_write_bytecode = True 
# discretization parameters:
nfe, ncp = 24, 3 # Radau nodes assumed in code
# state variables:
x_vars = {"PO":[()], "Y":[()], "W":[()], "PO_fed":[()], "MY":[()], "MX":[(0,),(1,)], "T":[()], "T_cw":[()]}
# corrupted state variables:
x_noisy = []
# output variables:
y_vars = {"PO":[()], "Y":[()], "MY":[()], 'T':[()], 'T_cw':[()]}
# controls:
u = ["u1", "u2"]
u_bounds = {"u1": (0.0,0.4), "u2": (0.0, 3.0)} 
# uncertain parameters:
p_noisy = {"A":[('p',),('i',)],'kA':[()]}
# noisy initial conditions:
noisy_ics = {'PO_ic':[()],'T_ic':[()],'MY_ic':[()],'MX_ic':[(1,)]}
# initial uncertainty set description (hyperrectangular):
p_bounds = {('A', ('i',)):(-0.2,0.2),('A', ('p',)):(-0.2,0.2),('kA',()):(-0.2,0.2),
            ('PO_ic',()):(-0.02,0.02),('T_ic',()):(-0.005,0.005),
            ('MY_ic',()):(-0.01,0.01),('MX_ic',(1,)):(-0.002,0.002)}
# time horizon bounds:
tf_bounds = [10.0*24/nfe, 30.0*24/nfe]
# path constrained properties to be monitored:
pc = ['Tad','T']
# monitored vars:
poi = [x for x in x_vars] + u
#parameter scenario:
scenario = {('A',('p',)):-0.2,('A',('i',)):-0.2,('kA',()):-0.2}
# scenario-tree definition:
st = {} # scenario tree : {parent_node, scenario_number on current stage, base node (True/False), scenario values {'name',(index):value}}
s_max, nr, alpha = 9, 1, 0.2
for i in range(1,nfe+1):
    if i < nr + 1:
        for s in range(1,s_max**i+1):
            if s%s_max == 1:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),True,{('A',('p',)):1.0,('A',('i',)):1.0,('kA',()):1.0}) 
            elif s%s_max == 2:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
            elif s%s_max == 3:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0-alpha})
            elif s%s_max == 4:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
            elif s%s_max == 5:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0-alpha})
            elif s%s_max == 6:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
            elif s%s_max == 7:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0+alpha,('kA',()):1.0+alpha})
            elif s%s_max == 8:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0+alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
            else:
                st[(i,s)] = (i-1,int(np.ceil(s/float(s_max))),False,{('A',('p',)):1.0-alpha,('A',('i',)):1.0-alpha,('kA',()):1.0+alpha})
    else:
        for s in range(1,s_max**nr+1):
            st[(i,s)] = (i-1,s,True,st[(i-1,s)][3])
sr = s_max**nr

# create MHE-NMPC-controller object
cl = msMHEGen(d_mod = SemiBatchPolymerization_multistage,
           d_mod_mhe = SemiBatchPolymerization,
           y=y_vars,
           x=x_vars,           
           x_noisy=x_noisy,
           p_noisy=p_noisy,
           u=u,
           u_bounds = u_bounds,
           tf_bounds = tf_bounds,
           poi = x_vars,
           scenario_tree = st,
           robust_horizon = nr,
           s_max = sr,
           noisy_inputs = False,
           noisy_params = True,
           adapt_params = True,
           update_scenario_tree = True,
           process_noise_model = None,#'params_bias',
           uncertainty_set = p_bounds,
           confidence_threshold = alpha,
           robustness_threshold = 0.05,
           obj_type='economic',
           control_displacement_penalty=True,
           nfe_t=nfe,
           ncp_t=ncp,
           path_constraints=pc)

cov_matrices = {'y_cov':mcov,'q_cov':qcov,'u_cov':ucov,'p_cov':pcov}
reg_weights = {'K_w':1.0}
stgen_in = {'epc':['PO_ptg','mw','unsat','mw_ub'],'pc':['T_max','T_min','temp_b'],'noisy_ics':noisy_ics,'par_bounds':p_bounds}

args = {'cov_matrices':cov_matrices, 'regularization_weights':reg_weights,'stgen_args':stgen_in,
        'advanced_step':False,'fix_noise':True,'stgen':False,'meas_noise':x_measurement,'disturbance_src':{}}

kind = 'msNMPC'
parest = '_parest_' if cl.adapt_params else '_'
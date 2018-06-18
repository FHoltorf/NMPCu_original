#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:52:39 2017

@author: flemmingholtorf
"""
#### 
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix, ConstraintList
from pyomo.core.base.sets import SimpleSet
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from main.dync.NMPCGen import NMPCGen, msNMPCGen
from itertools import product
from copy import deepcopy
from scipy.stats import chi2
import sys, time, resource
import numpy as np

__author__ = "@FHoltorf"

""" a strong base is essential """
class MHEGen(NMPCGen):
    def __init__(self, **kwargs):
        NMPCGen.__init__(self, **kwargs)
        self.int_file_mhe_suf = int(time.time())-1

        self.y = kwargs.pop('y', {})

        self.x_noisy = kwargs.pop('x_noisy', [])
        self.noisy_ics = kwargs.pop('noisy_ics', True)
        self.diag_Q_R = kwargs.pop('diag_QR', True)  #: By default use diagonal matrices for Q and R matrices
        self.u = kwargs.pop('u', [])
        self.initial_confidence = kwargs.pop('initial_confidence',1e-6)
        
        self.measurement = {}
        self.noisy_controls = kwargs.pop('noisy_controls', False)
        self.p_noisy = kwargs.pop('p_noisy', {})
        self.process_noise_model = kwargs.pop('process_noise_model',None)
        self.noisy_params = kwargs.pop('noisy_params', False)                
        self.d_mod_mhe = kwargs.pop('d_mod_mhe', self.d_mod)
        self.update_uncertainty_set = kwargs.pop('update_uncertainty_set', False)
        self.mhe_confidence_ellipsoids = {}
        
    def create_mhe(self):
        self.lsmhe = self.d_mod_mhe(self.nfe_mhe, self.ncp_t)
        self.lsmhe.name = "lsmhe (Least-Squares MHE)"
        self.lsmhe.create_bounds()
        #self.lsmhe.clear_aux_bounds()
        self.lsmhe.clear_all_bounds()
        self.lsmhe.create_output_relations()
        
        #: Create list of noisy-states vars
        self.xkN_l = []
        self.xkN_nexcl = []
        self.xkN_key = {}
        k = 0
        for x in self.x_noisy:
            n_s = getattr(self.lsmhe, x)  #: Noisy-state
            for jth in self.x[x]:  #: the jth variable
                self.xkN_l.append(n_s[(1, 0) + jth])
                self.xkN_nexcl.append(1)  #: non-exclusion list for active bounds
                self.xkN_key[(x, jth)] = k
                k += 1

        self.lsmhe.xkNk_mhe = Set(initialize=[i for i in range(0, len(self.xkN_l))])  #: Create set of noisy_states
        self.lsmhe.wk_mhe = Var(range(0,self.nfe_mhe), self.lsmhe.xkNk_mhe, initialize=0.0)
        self.lsmhe.Q_mhe = Param(range(0, self.nfe_mhe), self.lsmhe.xkNk_mhe, initialize=1, mutable=True) if self.diag_Q_R\
            else Param(range(0, self.nfe_mhe), self.lsmhe.xkNk_mhe, self.lsmhe.xkNk_mhe,
                             initialize=lambda m, t, i, ii: 1.0 if i == ii else 0.0, mutable=True)  #: Disturbance-weight
        
        # deactivates the continuity constraints in collocation scheme
        for i in self.x_noisy:
            cp_con = getattr(self.lsmhe, "cp_" + i)
            cp_con.deactivate()
        
        #: Create list of measurements vars
        self.yk_l = {}
        self.yk_key = {}
        k = 0
        self.yk_l[1] = []
        for y in self.y:
            m_v = getattr(self.lsmhe, y)  #: Measured "state"
            for jth in self.y[y]:  #: the jth variable
                self.yk_l[1].append(m_v[(1, self.ncp_t) + jth])
                self.yk_key[(y, jth)] = k  #: The key needs to be created only once, that is why the loop was split
                k += 1

        for t in range(2, self.nfe_mhe + 1):
            self.yk_l[t] = []
            for y in self.y:
                m_v = getattr(self.lsmhe, y)  #: Measured "state"
                for jth in self.y[y]:  #: the jth variable
                    self.yk_l[t].append(m_v[(t, self.ncp_t) + jth])

        self.lsmhe.ykk_mhe = Set(initialize=[i for i in range(0, len(self.yk_l[1]))])  #: Create set of measured_vars
        self.lsmhe.nuk_mhe = Var(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=0.0)   #: Measurement noise
        self.lsmhe.yk0_mhe = Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=1.0, mutable=True) #:
    
        # creates constraints that quantify the respective measurement noise
        self.lsmhe.hyk_c_mhe = Constraint(self.lsmhe.fe_t, self.lsmhe.ykk_mhe,
                                          rule=
                                          lambda mod, t, i:mod.yk0_mhe[t, i] - self.yk_l[t][i] - mod.nuk_mhe[t, i] == 0.0)
        self.lsmhe.hyk_c_mhe.deactivate()
        # creates  Identity matrix for weighting of the measurement noise
        self.lsmhe.R_mhe = Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=1.0, mutable=True) if self.diag_Q_R else \
            Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, self.lsmhe.ykk_mhe,
                             initialize=lambda mod, t, i, ii: 1.0 if i == ii else 0.0, mutable=True)
        f = open("file_cv.txt", "w")
        f.close()
        if self.noisy_controls:
            for u in self.u:
                self.lsmhe.add_component("w_" + u + "_mhe", Var(self.lsmhe.fe_t, initialize=0.0))  #: Noise for input vars
                self.lsmhe.add_component("w_" + u + "c_mhe", Constraint(self.lsmhe.fe_t))
                self.lsmhe.equalize_u(direction="r_to_u")
                
                cc = getattr(self.lsmhe, u + "_c")  #: Get the constraint for input
                cc.deactivate()
                
                con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
                wu = getattr(self.lsmhe, "w_" + u + "_mhe")  #: Get the constraint-noisy
                ce = getattr(self.lsmhe, u + "_e")  #: Get the expression
                cp = getattr(self.lsmhe, u)  #: Get the param
    
                con_w.rule = lambda m, i: cp[i] == ce[i] + wu[i]
                con_w.reconstruct()
                con_w.deactivate()

        # initialize as identitiy --> U_mhe = weighting/inv(covariance) of the control noise
        self.lsmhe.U_mhe = Param(self.lsmhe.fe_t, self.u, initialize=1, mutable=True)

        #: Deactivate icc constraints
        if self.noisy_ics:
            # for semi-batch need to introduce also penalty for the initial conditions
            self.lsmhe.noisy_ic = ConstraintList()
            for i in self.x_noisy:
                    ic_con = getattr(self.lsmhe,i + "_icc")
                    ic_exp = getattr(self.lsmhe,i + "_ice")
          
                    # set initial guess
                    xic = getattr(self.lsmhe,i + "_ic")
                    x = getattr(self.lsmhe,i)
                    for j in self.x[i]:
                        k = self.xkN_key[(i,j)] # key that belongs to the certain variable for wk_mhe
                        if j == ():
                            x[(1,0)+j] = xic.value # set reasonable initial guess
                            self.lsmhe.noisy_ic.add(ic_exp == self.lsmhe.wk_mhe[0,k]) # add noisy initial condition
                        else:
                            x[(1,0)+j] = xic[j].value
                            self.lsmhe.noisy_ic.add(ic_exp[j] == self.lsmhe.wk_mhe[0,k]) # add noisy initial condition
                        
                    for k in ic_con.keys():
                        ic_con[k].deactivate() # deactivate the old constraints
        else:
            pass
        
        #: Put the noise in the continuation equations (finite-element)
        j = 0
        self.lsmhe.noisy_cont = ConstraintList()
        for x in self.x_noisy:
            cp_exp = getattr(self.lsmhe, "noisy_" + x)
            for key in self.x[x]:  #: This should keep the same order
                for t in range(1, self.nfe_mhe):
                    self.lsmhe.noisy_cont.add(cp_exp[t, key] == self.lsmhe.wk_mhe[t, j])
                j += 1
         
        self.lsmhe.noisy_cont.deactivate()
       
        self.lsmhe.Q_e_mhe = Expression(
            expr= 1.0/2.0 * sum(
                sum(
                    self.lsmhe.Q_mhe[i, k] * self.lsmhe.wk_mhe[i, k]**2 for k in self.lsmhe.xkNk_mhe)
                for i in range(0, self.nfe_mhe))) if self.diag_Q_R else Expression(
            expr=sum(sum(self.lsmhe.wk_mhe[i, j] *
                         sum(self.lsmhe.Q_mhe[i, j, k] * self.lsmhe.wk_mhe[i, k] for k in self.lsmhe.xkNk_mhe)
                         for j in self.lsmhe.xkNk_mhe) for i in range(0, self.nfe_mhe)))

        self.lsmhe.R_e_mhe = Expression(
            expr=1.0/2.0 * sum(
                sum(
                    self.lsmhe.R_mhe[i, k] * self.lsmhe.nuk_mhe[i, k]**2 for k in self.lsmhe.ykk_mhe)
                for i in self.lsmhe.fe_t)) if self.diag_Q_R else Expression(
            expr=sum(sum(self.lsmhe.nuk_mhe[i, j] *
                         sum(self.lsmhe.R_mhe[i, j, k] * self.lsmhe.nuk_mhe[i, k] for k in self.lsmhe.ykk_mhe)
                         for j in self.lsmhe.ykk_mhe) for i in self.lsmhe.fe_t))
        
        # process_noise_model == time-variant paramters
        self.lsmhe.P_e_mhe = Expression(expr= 0.0)
        if self.process_noise_model == 'params':
            self.pkN_l = []
            self.pkN_nexcl = []
            self.pkN_key = {}
            k = 0
            for p in self.p_noisy:
                try:# check whether parameter is time-variant
                    par = getattr(self.lsmhe, 'p_' + p)
                except:# catch time-invariant parameters
                    self.lsmhe.par_to_var()
                    par = getattr(self.lsmhe, p)
                    par.unfix()
                    print('Parameter ' + p + ' is not time-variant')
                    continue
                for jth in self.p_noisy[p]:  #: the jth variable
                    self.pkN_l.append(par[(1,) + jth])
                    self.pkN_nexcl.append(1)  #: non-exclusion list for active bounds
                    self.pkN_key[(p, jth)] = k
                    k += 1
    
            self.lsmhe.pkNk_mhe = Set(initialize=[i for i in range(0, len(self.pkN_l))])  #: Create set of noisy_states
            self.lsmhe.dk_mhe = Var(self.lsmhe.fe_t, self.lsmhe.pkNk_mhe, initialize=0.0, bounds=(-0.99,1.0))
            self.lsmhe.P_mhe = Param(self.lsmhe.pkNk_mhe, initialize=1.0, mutable=True)
            self.lsmhe.noisy_pars = ConstraintList()
            
            for p in self.p_noisy:
                try:# check whether parameter is time-variant
                    par = getattr(self.lsmhe, 'p_' + p)
                except:# catch time-invariant parameters
                    continue
                for key in self.p_noisy[p]:
                    j = self.pkN_key[(p,key)]
                    for t in range(1,self.nfe_mhe + 1):
                        self.lsmhe.noisy_pars.add(par[(t,)+key] - 1.0 - self.lsmhe.dk_mhe[t,j] == 0.0)
                        par[(t,)+key].unfix()
 
                    
            self.lsmhe.P_e_mhe.expr = 1.0/2.0 * sum(sum(self.lsmhe.P_mhe[k] * self.lsmhe.dk_mhe[i, k]**2 \
                                                        for k in self.lsmhe.pkNk_mhe) \
                                                    for i in self.lsmhe.fe_t)
        elif self.process_noise_model == 'params_bias':
            self.pkN_l = []
            self.pkN_nexcl = []
            self.pkN_key = {}
            k = 0
            for p in self.p_noisy:
                try:# check whether parameter is time-variant
                    par = getattr(self.lsmhe, 'p_' + p)
                except:# catch time-invariant parameters
                    print('Parameter ' + p + ' is not time-variant')
                    continue
                for jth in self.p_noisy[p]:  #: the jth variable
                    self.pkN_l.append(par[(1,) + jth])
                    self.pkN_nexcl.append(1)  #: non-exclusion list for active bounds
                    self.pkN_key[(p, jth)] = k
                    k += 1
    
            self.lsmhe.pkNk_mhe = Set(initialize=[i for i in range(0, len(self.pkN_l))])  #: Create set of noisy_states
            self.lsmhe.dk_mhe = Var(self.lsmhe.fe_t, self.lsmhe.pkNk_mhe, initialize=0.0, bounds=(-0.5,0.5))
            self.lsmhe.P_mhe = Param(self.lsmhe.pkNk_mhe, initialize=1.0, mutable=True)
            self.lsmhe.noisy_pars = ConstraintList()
            
            for p in self.p_noisy:
                try:# check whether parameter is time-variant
                    par = getattr(self.lsmhe, 'p_' + p)
                except:# catch time-invariant parameters
                    continue
                for key in self.p_noisy[p]:
                    j = self.pkN_key[(p,key)]
                    for t in range(1,self.nfe_mhe + 1):
                        jth = (t,)+key
                        self.lsmhe.noisy_pars.add(par[jth] - 1.0 - self.lsmhe.dk_mhe[t,j] == 0.0)
                        par[jth].unfix()

            self.lsmhe.P_e_mhe.expr = 1.0/2.0 * (sum(self.lsmhe.P_mhe[k] * self.initial_confidence * (self.lsmhe.dk_mhe[1, k]-0.0)**2.0 for k in self.lsmhe.pkNk_mhe) +\
                                                sum(sum(self.lsmhe.P_mhe[k] * (self.lsmhe.dk_mhe[i, k]-self.lsmhe.dk_mhe[i-1,k])**2 \
                                                for k in self.lsmhe.pkNk_mhe) for i in range(2,self.nfe_mhe+1)))
            
        else:
            pass # no process noise
            
        expr_u_obf = 0
        
        if self.noisy_controls:
            for i in self.lsmhe.fe_t:
                for u in self.u:
                    var_w = getattr(self.lsmhe, "w_" + u + "_mhe")  #: Get the constraint-noisy
                    expr_u_obf += self.lsmhe.U_mhe[i, u] * var_w[i] ** 2

        self.lsmhe.U_e_mhe = Expression(expr= 1.0/2.0 * expr_u_obf)  # how about this

        self.lsmhe.Arrival_e_mhe = Expression(expr = 0.0)

        self.lsmhe.obfun_dum_mhe_deb = Objective(sense=minimize,
                                             expr=1.0)
        self.lsmhe.obfun_dum_mhe_deb.deactivate()
        
        self.lsmhe.obfun_dum_mhe = Objective(sense=minimize,
                                             expr=self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe + self.lsmhe.P_e_mhe)
        self.lsmhe.obfun_dum_mhe.deactivate()

        self.lsmhe.obfun_mhe = Objective(sense=minimize,
                                         expr=self.lsmhe.Arrival_e_mhe + self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe + self.lsmhe.P_e_mhe)
        self.lsmhe.obfun_mhe.deactivate()


        # deactivate endpoint constraints
        self.lsmhe.deactivate_epc()
        self.lsmhe.deactivate_pc()

        self._PI = {}  #: Container for the reduced Hessian
        self.xreal_W = {}
        self.curr_m_noise = {}   #: Current measurement noise
        self.curr_y_offset = {}  #: Current offset of measurement
        for y in self.y:
            for j in self.y[y]:
                self.curr_m_noise[(y, j)] = 0.0
                self.curr_y_offset[(y, j)] = 0.0

        self.s_estimate = {}
        self.s_real = {}
        for x in self.x_noisy:
            self.s_estimate[x] = []
            self.s_real[x] = []

        self.y_estimate = {}
        self.y_real = {}
        self.y_noise_jrnl = {}
        self.yk0_jrnl = {}
        for y in self.y:
            self.y_estimate[y] = []
            self.y_real[y] = []
            self.y_noise_jrnl[y] = []
            self.yk0_jrnl[y] = []        
    
                
    def cycle_mhe(self,**kwargs):
        m = self.lsmhe if self._c_it > 1 else self.recipe_optimization_model
        initialguess = kwargs.pop('init',self.store_results(m))
        
        self.create_mhe()
        
        # open the parameters as degree of freedom
        if self.noisy_params: 
            self.lsmhe.par_to_var()
            for p in self.p_noisy:
                p_mhe = getattr(self.lsmhe,p)
                for key in self.p_noisy[p]:
                    pkey = None if key == () else key
                    p_mhe[pkey].unfix()
        
        # load initialguess from previous iteration lsmhe and olnmpc for the missing element
        if self._c_it == 1:
            for var in self.lsmhe.component_objects(Var):
                for key in var.index_set():
                    try:
                        var[key].value = initialguess[(var.name,key)]
                    except KeyError:
                        try:
                            if type(key) == int:
                                aux_key = (key,)
                            elif key == None:
                                continue
                            else:
                                aux_key = (1,) + key[1:]
                            var[key].value = initialguess[(var.name,aux_key)] # xxx
                        except KeyError: # fallback strategy
                            continue
                        except AttributeError: # fallback strategy
                            continue
        else:
            for var in self.lsmhe.component_objects(Var):
                for key in var.index_set():
                    try:
                        var[key].value = initialguess[(var.name,key)]
                    except KeyError:
                        try:
                            var_nmpc = getattr(self.olnmpc, var.name)
                            if type(key) == int:
                                aux_key = (key,)
                            elif key == None:
                                continue
                            else:
                                aux_key = (1,) + key[1:]
                            var[key].value = var_nmpc[aux_key].value # xxx
                        except KeyError: # fallback strategy
                            continue
                        except AttributeError: # fallback strategy
                            continue
                
            
        # adjust the time intervals via model parameter self.lsmhe.fe_dist[i]
        self.lsmhe.tf.fix(self.recipe_optimization_model.tf.value) # base is set via recipe_optimization_model
        for i in self.lsmhe.fe_t:
            self.lsmhe.fe_dist[i] = (self.nmpc_trajectory[i,'tf']-self.nmpc_trajectory[i-1,'tf'])/self.lsmhe.tf.value
        
        # fix the applied controls in the model:    
        # in case EVM is used --> slacks are added and penalized
        for u in self.u:
            control = getattr(self.lsmhe, u)
            control[self.nfe_mhe].value = self.curr_u[u]
            control.fix()
        
        self.lsmhe.equalize_u()
        

        
        # activate whats necessary + leggo:
        self.lsmhe.obfun_mhe.activate() # objective function!
        self.lsmhe.noisy_cont.activate() # noisy constraints!
        self.lsmhe.hyk_c_mhe.activate()
        self.lsmhe.eobj.deactivate()
        if self.noisy_controls:
            for u in self.u:
                con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
                con_w.activate()

    def set_measurement_prediction(self,y_cov = {}, q_cov = {}, u_cov = {}, p_cov = {}, **kwargs):
        results = kwargs.pop('prediction',self.store_results(self.forward_simulation_model))
        # needs to be called for construction of lsmhe
        # however only of importance when asMHE is applied
        measured_state = {}
        for x in self.x:
            for j in self.x[x]:
                measured_state[(x,j)] = results[(x,(1,3)+j)]
        for y in self.y:
            for j in self.y[y]:
                measured_state[(y,j)] = results[(y,(1,3)+j)]
        self.measurement[self._c_it] = deepcopy(measured_state)
    
                # reapply the measured variables
        for t in self.lsmhe.fe_t:
            for y in self.y:
                for j in self.y[y]:
                    vni = self.yk_key[(y,j)]
                    self.lsmhe.yk0_mhe[t,vni] = self.measurement[t][(y,j)]
                    
        # redo the initialization of estimates and matrices
        self.set_covariance_y(y_cov)
        self.set_covariance_w(q_cov)
        if self.noisy_controls:
            self.set_covariance_u(u_cov)
        if self.process_noise_model == 'params' \
           or self.process_noise_model == 'params_bias':
            self.set_covariance_p(p_cov)
            
    def create_measurement(self,var_dict, y_cov = {}, q_cov = {}, u_cov = {}, p_cov = {}):
        """ Args:
                var_dict = {(x.name,(add_indices)):relative variance for measurement noise}
        """
        # creates measurement for all variables that are in var_dict        
        # prediction for these variables must have been generated beforehand 
        
        # Sets possibly more measured states than are loaded into the state estimation problem
        # what is specified in self.y determines which variables are considered as measured for estimation
        measured_state = {}
        for key in var_dict:
                yvar = getattr(self.plant_simulation_model, key[0])
                j = key[1]
                sigma = var_dict[key]
                rand = np.random.normal(loc=0.0, scale=sigma)
                if abs(rand) > 2*sigma:
                    rand = -2*sigma if rand < 0.0 else 2*sigma
                else:
                    pass
                measured_state[key] = (1-rand)*yvar[(1,3)+j].value
        self.measurement[self._c_it] = deepcopy(measured_state)
        
        # update values in lsmhe problem
        for t in self.lsmhe.fe_t:
            for y in self.y:
                for j in self.y[y]:
                    vni = self.yk_key[(y,j)]
                    self.lsmhe.yk0_mhe[t,vni] = self.measurement[t][(y,j)]
                    
        # redo the initialization of estimates and matrices
        self.set_covariance_y(y_cov)
        self.set_covariance_w(q_cov)
        if self.noisy_controls:
            self.set_covariance_u(u_cov)
        if self.process_noise_model == 'params' \
           or self.process_noise_model == 'params_bias':
            self.set_covariance_p(p_cov)
            
    def solve_mhe(self,fix_noise=False):
        self.journalizer('U', self._c_it, 'state estimation problem','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        if fix_noise:
            self.lsmhe.wk_mhe.fix()
        
        initialguess = self.store_results(self.lsmhe)
        #for var in self.lsmhe.component_objects(Var):
        #        for key in var.index_set():
        #            var[key].value = initialguess[var.name,key]   
                    
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "no"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 500
        with open("ipopt.opt", "w") as f:
            f.write("print_info_string yes")
            f.close()


        result = ip.solve(self.lsmhe, tee=True)
        
        if [str(result.solver.status),str(result.solver.termination_condition)] != ['ok','optimal']:
            for p in self.p_noisy:
                par_mhe = getattr(self.lsmhe, p)
                par_true = getattr(self.plant_simulation_model, p)
                for key in self.p_noisy[p]:
                    pkey = None if key ==() else key
                    print('estimated')
                    par_mhe.pprint()
                    print('true')
                    par_true.pprint()
            for var in self.lsmhe.component_objects(Var):
                for key in var.index_set():
                    var[key].value = initialguess[var.name,key]        
            self.lsmhe.create_bounds()
            self.lsmhe.clear_aux_bounds()
            self.lsmhe.par_to_var()
            result = ip.solve(self.lsmhe, tee=True)
        
        # saves the results to initialize the upcoming mhe problem
        # is up to the user to do it by hand
        output = self.store_results(self.lsmhe) # saves the results to initialize the upcoming mhe problem
    
        if self.adapt_params:
            # {timestep after which the parameters were identified, 'e_pars':dictionary with estimates}
            for p in self.p_noisy:
                p_mhe = getattr(self.lsmhe,p)
                for key in self.p_noisy[p]:
                    pkey = key if key != () else None 
                    self.curr_epars[(p,key)] = p_mhe[pkey].value
                    
            self.nmpc_trajectory[self._c_it,'e_pars'] = deepcopy(self.curr_epars)
            
        # saves the predicted initial_values for the states
        for x in self.x:
            for j in self.x[x]:
                self.initial_values[(x,j)] = output[(x,(self.nfe_mhe,3)+j)]
                
        # load solution status in self.nmpc_trajectory
        self.nmpc_trajectory[self._c_it,'solstat_mhe'] = [str(result.solver.status),str(result.solver.termination_condition)]
        self.nmpc_trajectory[self._c_it,'obj_value_mhe'] = value(self.lsmhe.obfun_mhe)
        
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
        
    def cycle_ics_mhe(self, nmpc_as = False, mhe_as = False, alpha=0.95):
        if nmpc_as and mhe_as: # both nmpc and mhe use advanced step scheme
            ivs = self.initial_values
        elif nmpc_as and not(mhe_as): # only nmpc uses advanced step scheme
            ivs = self.curr_pstate
        else: # nothing uses advanced step scheme
            ivs = self.initial_values
            
        for x in self.x:
            xic = getattr(self.olnmpc, x+'_ic')
            for j in self.x[x]:
                if not(j == ()):
                    xic[j].value = ivs[(x,j)]
                else:
                    xic.value = ivs[(x,j)]

        if (self.adapt_params or self.update_uncertainty_set) \
            and self._c_it > 1:
            ###################################################################
            # comute principle components of approximate 95%-confidence region 
            ###################################################################
            try:
                dimension = int(np.round(np.sqrt(len(self.mhe_confidence_ellipsoids[1]))))
                confidence = chi2.isf(1-alpha,dimension)
                A_dict = self.mhe_confidence_ellipsoids[self._c_it-1]
                # assemble dimension x dimension array
                rows = {}
                for m in range(dimension):
                        rows[m] = np.array([A_dict[(m,i)] for i in range(dimension)]) 
                A = 1/confidence*np.array([np.array(rows[i]) for i in range(dimension)]) # shape matrix of confidence ellipsoid, should rewrite and scale according to parameter values, much simpler afterwards
                U, s, V = np.linalg.svd(A) # singular value decomposition of shape matrix 
 
                # 1/sqrt since A = 1/xi^2 * V_p^-1 and we look for the set that satisfies
                # 1/xi^2 * x^T V_p^-1 x <= 1 <=> x^T A x <= 1 <=> ||A^1/2 x||_2 <= 1
                # accordingly radii are provided by inverse of sqrt of singular values of A
                radii = 1/np.sqrt(s) # radii -- 
                
                # adapt parameters iff estimates are confident enough
                if self.adapt_params:
                    for p in self.p_noisy:
                        p_nom = getattr(self.olnmpc,p)
                        p_mhe = getattr(self.lsmhe,p)
                        for key in self.p_noisy[p]:                           
                            pkey = None if key == () else key
                            index = self.PI_indices[p,key]
                            dev = -1e8
                            for m in range(dimension):
                                 dev = max(dev,(abs(radii[m]*U[index][m]) + p_mhe[pkey].value)/p_mhe[pkey].value)
                            if dev < 1 + self.estimate_acceptance:
                                self.curr_epars[(p,key)] = p_mhe[pkey].value
                            else:
                                self.curr_epars[(p,key)] = p_nom[pkey].value
                                
            except KeyError: # adapt parameters blindly
                if self.adapt_params:
                    for p in self.p_noisy:
                        p_mhe = getattr(self.lsmhe,p)
                        for key in self.p_noisy[p]:
                            pkey = None if key == () else key
                            self.curr_epars[(p,key)] = p_mhe[pkey].value
            
            
            ###############################################################
            ###############################################################
            # alternative: use the projections on the axis
            # projection x^T A x = 1
            # A_k = A with kth row and kth column with zeros and akk = 1
            # solve A_k*D_k = Z_k 
            # with Z_k = -a_k (kth column of A) and Z_k,k = 1.0
            # delta_x_k = np.sqrt(a_k^T*D_k)
            
            if self.update_uncertainty_set:
                dev = {(p,key):self.confidence_threshold for p in self.p_noisy for key in self.p_noisy[p]}
                for p in self.p_noisy:
                    p_mhe = getattr(self.lsmhe,p)
                    for key in self.p_noisy[p]:
                        pkey = key if key != () else None
                        k = self.PI_indices[p,key]
                        A_k = deepcopy(A)
                        A_k[:,k] = 0.0
                        A_k[k,:] = 0.0
                        A_k[k,k] = 1.0
                        Z_k = np.zeros(dimension)
                        Z_k[:] = -A[:,k]
                        Z_k[k] = 1.0
                        a_k = A[k,:]
                        D_k = np.linalg.solve(A_k,Z_k)
                        dev_k = np.sqrt(1/np.dot(a_k,D_k))/p_mhe[pkey].value
                        # use new interval if robustness_threshold < dev_k < confidence_threshold
                        dev[(p,key)] = min(self.confidence_threshold,max(self.robustness_threshold, dev_k))

                # create new weighting matrix:
                for p in self.p_noisy:
                    for key in self.p_noisy[p]:
                        self.alpha[(p,key)] = dev[(p,key)]
                        
    def set_covariance_y(self, cov_dict):
        """Sets covariance(inverse) for the measurements.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(meas_name, j), (meas_name, k), time]
        Returns:
            None
        """
        # measured states R_mhe
        rtarget = getattr(self.lsmhe, "R_mhe")
        for key in cov_dict if self.diag_Q_R else cov_dict[0]: 
            # does hardly ever make sense to have time-dependent measuremet noise
            # 1. check whether the variable is even a measured one
            if key[0] in self.yk_key: 
                # 2. if yes: compute the respective weight as follows:
                vni = key[0]
                vnj = key[1]
                v_i = self.yk_key[vni]
                v_j = self.yk_key[vnj]
                for t in range(1,self.nfe_mhe+1):
                    if self.diag_Q_R:                
                        rtarget[t, v_i] = 1 / max(abs(cov_dict[vni, vnj]*self.measurement[t][vni]),1e-4)**2
                    else:
                        # only allow for diagonal measurement covariance matrices 
                        rtarget[t, v_i, v_j] = 1 / max(abs(cov_dict[vni, vnj]*self.measurement[t][vni]),1e-4)**2
            else:
                continue

    def set_covariance_w(self, cov_dict, set_bounds=True):
        """Sets covariance(inverse) for the states.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(state_name, j), (state_name, k), time]
        Returns:
            None
        """
        # disturbances
        qtarget = getattr(self.lsmhe, "Q_mhe")
        w = getattr(self.lsmhe, 'wk_mhe')
        for key in cov_dict if self.diag_Q_R else cov_dict[0]:
            if key[0] in self.xkN_key:
                vni = key[0]
                vnj = key[1]
                v_i = self.xkN_key[vni]
                v_j = self.xkN_key[vnj]
                xic = getattr(self.lsmhe, vni[0] + "_ic")
                xkey = None if vni[1] == () else vni[1]
                if self.diag_Q_R:
                    qtarget[0, v_i] = 1.0 / (max(abs(cov_dict[vni, vnj]*xic[xkey].value),1e-4))**2 # .0001
                    if set_bounds:
                        if cov_dict[0][vni, vnj] != 0:
                            confidence = 3*cov_dict[0][vni, vnj]*xic[xkey].value
                            w[0,v_i].setlb(-confidence)
                            w[0,v_i].setub(confidence)
                            if abs(w[0,v_i].lb - w[0,v_i].ub) < 1e-3:
                                w[0,v_i].fix(0.0)
                        else:
                            w[0,v_i].fix(0.0)
                else:
                    qtarget[0, v_i, v_j] = cov_dict[0][vni, vnj]                     
                for t in range(1,self.nfe_mhe):
                    if self.diag_Q_R:
                        qtarget[t, v_i] = 1 / (max(abs(cov_dict[vni, vnj]*self.nmpc_trajectory[t,vni]),1e-4))**2 # 0.00001
                        if set_bounds:
                            if cov_dict[vni, vnj] != 0.0:
                                    confidence = 10*abs(cov_dict[vni,vnj]*self.nmpc_trajectory[t,vni])
                            else:
                                confidence = 0.0 
                            w[t,v_i].setlb(-confidence)
                            w[t,v_i].setub(confidence)
                            if abs(w[t,v_i].lb - w[t,v_i].ub) < 1e-3:
                                w[t,v_i].fix()
                    else:
                        if (vni,vnj) in cov_dict[t]:
                            qtarget[t, v_i, v_j] = cov_dict[t][vni, vnj]
                        else:
                            print('no weight for element : \n',vni,'\n',vnj,'\n specified')
                            qtarget[t, v_i, v_j] = 1e8
            else:
                print(key[0], ' is not a noisy state variable')

    def set_covariance_u(self, cov_dict):
        """Sets covariance(inverse) for the states.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(state_name, j), (state_name, k), time]
        Returns:
            None
        """
        utarget = getattr(self.lsmhe, "U_mhe")
        for key in cov_dict:
            vni = key[0]
            for _t in range(1,self.nfe_mhe+1):
                aux_key = self.nfe_t_0 if _t > self.nfe_t_0 else _t
                utarget[_t, vni] = 1 / (max(abs(cov_dict[key]*self.reference_control_profile[vni,aux_key]),1e-4))**2
    
    def set_covariance_p(self, cov_dict, set_bounds=True):
        ptarget = getattr(self.lsmhe, "P_mhe")
        for key in cov_dict:
            # only
            p = key[0][0]
            index = key[0][1]
            k = self.pkN_key[(p,index)]
            ptarget[k] = 1/cov_dict[key]**2.0

    def compute_measurement_offset(self):
        mhe_y = getattr(self.lsmhe, "yk0_mhe")
        for y in self.y:
            for j in self.y[y]:
                k = self.yk_key[(y, j)]
                mhe_yval = value(mhe_y[self.nfe_mhe, k])
                self.curr_y_offset[(y, j)] = mhe_yval - self.measurement[self._c_it][(y,j)]           
    
    def compute_confidence_ellipsoid(self):
        """ computes confidence ellipsoids for estimated parameters via 1st order approximation"""
        self.journalizer('U', self._c_it, 'confidence ellipsoid computation','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name

        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        # set parameters
        for p in self.p_noisy:
            par = getattr(self.lsmhe,p) 
            for j in self.p_noisy[p]:
                if j == ():
                    par.set_suffix_value(self.lsmhe.dof_v, 1)
                else:
                    par[j].set_suffix_value(self.lsmhe.dof_v, 1)
        
        self.lsmhe.ipopt_zL_in.update(self.lsmhe.ipopt_zL_out)
        self.lsmhe.ipopt_zU_in.update(self.lsmhe.ipopt_zU_out)          

        self.journalizer("I", self._c_it, "load_covariance_prior", "K_AUG w red_hess")
        self.k_aug.options["compute_inv"] = ""
        self.k_aug.options["no_barrier"] = ""
        self.k_aug.options["no_scale"] = ""
        
        try:
            self.k_aug.solve(self.lsmhe, tee=True)
        except:
            self.nmpc_trajectory[self._c_it,'solstat_mhe'] = ['Inversion of Reduced Hessian failed','Inversion of Reduced Hessian failed']
      
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        self._PI.clear()
        self.PI_indices = {}
    
        for p in self.p_noisy:
            par = getattr(self.lsmhe,p) 
            for j in self.p_noisy[p]:
                if j == ():
                    aux = par.get_suffix_value(self.lsmhe.rh_name) 
                    if aux == None:
                        self.PI_indices[par.name,j] = 0
                    else:
                        self.PI_indices[par.name,j] = aux
                else:
                    aux = par[j].get_suffix_value(self.lsmhe.rh_name)
                    if aux == None:
                        self.PI_indices[par.name,j] = 0
                    else:
                        self.PI_indices[par.name,j] = aux
                   
        #"inv_.in" gives the reduced hessian which is the shape matrix in x^T A x = 1
        # inv_.in contains the reduced hessian
        # read from file and store in _PI according to order specified in _PI
        with open("inv_.in", "r") as rh:
            ll = []
            l = rh.readlines()
            row = 0
            for i in l:
                ll = i.split()
                col = 0
                for j in ll:
                    self._PI[row, col] = float(j)
                    col += 1
                row += 1
            rh.close()
        print("-" * 120)
        print("I[[load covariance]] e-states nrows {:d} ncols {:d}".format(len(l), len(ll)))
        print("-" * 120)
        
        # unscaled matrix
        self.mhe_confidence_ellipsoids[self._c_it] = deepcopy(self._PI)
        
        # scaled shape matrix of confidence ellipsoid based on the current estimate
        dim = len(self.PI_indices)
        S = np.zeros((dim,dim))
        confidence = chi2.isf(1.0-0.95,dim)
        rows = {}
        m = 0
        for p in self.PI_indices:
            p_mhe = getattr(self.lsmhe,p[0])
            key = p[1] if p[1] != () else None
            S[self.PI_indices[p]][self.PI_indices[p]] = p_mhe[key].value
            rows[m] = np.array([self._PI[(m,i)] for i in range(dim)])
            m += 1
        A = 1/confidence*np.array([np.array(rows[i]) for i in range(dim)]) 
        self._scaled_shape_matrix = np.dot(S.transpose(),np.dot(A,S)) # scaled shape matrix = scaled reduced hessian
        
        
        # alternatively much more efficient to just compute inverse of reduced hessian and scale that.
        # not is not required to compute the actual shape matrix of the ellipsoid
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
        
    """ for advanced-step MHE """
    def create_rh_sfx(self, set_suffix=True):
        """Creates relevant suffixes for k_aug (prior at fe=2) (Reduced_Hess)
        Args:
            set_suffix (bool): True if update must be done
        Returns:
            None
        """
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name

        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)

        if set_suffix:
            for key in self.x_noisy:
                var = getattr(self.lsmhe, key)
                for j in self.x[key]:
                    var[(2, 0) + j].set_suffix_value(self.lsmhe.dof_v, 1)

    def create_sens_suffix_mhe(self, set_suffix=True):
        """Creates relevant suffixes for k_aug (Sensitivity)
        Args:
            set_suffix (bool): True if update must be done
        Returns:
            None"""
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name
        if set_suffix:
            for key in self.x_noisy:
                var = getattr(self.lsmhe, key)
                for j in self.x[key]:
                    var[(self.nfe_mhe, self.ncp_t) + j].set_suffix_value(self.lsmhe.dof_v, 1)

    def compute_y_offset(self, noisy=True):
        mhe_y = getattr(self.lsmhe, "yk0_mhe")
        for y in self.y:
            plant_y = getattr(self.d1, y)
            for j in self.y[y]:
                k = self.yk_key[(y, j)]
                mhe_yval = value(mhe_y[self.nfe_mhe, k])
                plant_yval = value(plant_y[(1, self.ncp_t) + j])
                y_noise = self.curr_m_noise[(y, j)] if noisy else 0.0
                self.curr_y_offset[(y, j)] = mhe_yval - plant_yval - y_noise

    def sens_dot_mhe(self):
        """Updates suffixes, solves using the dot_driver"""
        self.journalizer("I", self._c_it, "sens_dot_mhe", "Set-up")

        if hasattr(self.lsmhe, "npdp"):
            self.lsmhe.npdp.clear()
        else:
            self.lsmhe.npdp = Suffix(direction=Suffix.EXPORT)
        self.create_sens_suffix_mhe()
        for y in self.y:
            for j in self.y[y]:
                k = self.yk_key[(y, j)]
                self.lsmhe.hyk_c_mhe[self.nfe_mhe, k].set_suffix_value(self.lsmhe.npdp, self.curr_y_offset[(y, j)])

        # with open("somefile0.txt", "w") as f:
        #     self.lsmhe.x.display(ostream=f)
        #     self.lsmhe.M.display(ostream=f)
        #     f.close()
        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        #: Looks for the file with the timestamp
        self.lsmhe.set_suffix_value(self.lsmhe.f_timestamp, self.int_file_mhe_suf)

        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        self.journalizer("I", self._c_it, "sens_dot_mhe", self.lsmhe.name)

        results = self.dot_driver.solve(self.lsmhe, tee=True, symbolic_solver_labels=True)
        self.lsmhe.solutions.load_from(results)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        # with open("somefile1.txt", "w") as f:
        #     self.lsmhe.x.display(ostream=f)
        #     self.lsmhe.M.display(ostream=f)
        #     f.close()

        ftiming = open("timings_dot_driver.txt", "r")
        s = ftiming.readline()
        ftiming.close()
        k = s.split()
        self._dot_timing = k[0]

    def sens_k_aug_mhe(self):
        self.journalizer("I", self._c_it, "sens_k_aug_mhe", "k_aug sensitivity")
        self.lsmhe.ipopt_zL_in.update(self.lsmhe.ipopt_zL_out)
        self.lsmhe.ipopt_zU_in.update(self.lsmhe.ipopt_zU_out)
        self.journalizer("I", self._c_it, "sens_k_aug_mhe", self.lsmhe.name)

        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        #: Now, the sensitivity step will have the timestamp for dot_in

        self.lsmhe.set_suffix_value(self.lsmhe.f_timestamp, self.int_file_mhe_suf)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)
        self.create_sens_suffix_mhe()
        results = self.k_aug_sens.solve(self.lsmhe, tee=True, symbolic_solver_labels=True)
        self.lsmhe.solutions.load_from(results)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)
        ftimings = open("timings_k_aug.txt", "r")
        s = ftimings.readline()
        ftimings.close()
        self._k_timing = s.split()

    def update_state_mhe(self, as_nmpc_mhe_strategy=False):
        # Improvised strategy
        if as_nmpc_mhe_strategy:
            self.journalizer("I", self._c_it, "update_state_mhe", "offset ready for asnmpcmhe")
            for x in self.x:
                xvar = getattr(self.lsmhe, x)
                xic = getattr(self.olnmpc, x + "_ic")
                for j in self.x[x]:
                    xkey = None if j == () else j
                    self.curr_state_offset[(x, j)] = value(xic[xkey])- value(xvar[self.nfe_mhe, self.ncp_t, j])
                    #print("state !", self.curr_state_offset[(x, j)])

        for x in self.x:
            xvar = getattr(self.lsmhe, x)
            for j in self.x[x]:
                self.curr_estate[(x, j)] = value(xvar[self.nfe_mhe, self.ncp_t, j])
            
                       
    """ MHE scheme """
    def run(self,meas_noise={}, cov_matrices={},regularization_weights={},disturbance_src={},\
            olrnmpc_args={}, advanced_step=False,fix_noise=False):
        ru = {}
        # off-line open-loop control
        ru['recipe_optimization']=self.recipe_optimization(**olrnmpc_args)
        self.set_reference_state_trajectory(self.get_state_trajectory(self.recipe_optimization_model))
        self.set_reference_control_profile(self.get_control_profile(self.recipe_optimization_model))
        
        self.create_nmpc() if self.obj_type == 'tacking' else self.create_enmpc() 
        self.load_reference_trajectories() if self.obj_type == 'tracking' else None
        
        self.create_mhe()

        # on-line control
        if advanced_step:
            for i in range(1,self.nfe_t_0):
                ru['plant_simulation',i]=self.plant_simulation(**disturbance_src)
                # preparation phase
                self.cycle_mhe() 
                self.cycle_nmpc() 
                if self.update_uncertainty_set:
                    ru['confidence_ellipsoid_computation',i]=self.compute_confidence_ellipsoid()
                self.cycle_ics_mhe(nmpc_as=advanced_step,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc
                self.load_reference_trajectories() if self.obj_type == 'tracking' else None # loads the reference trajectory in olnmpc problem (for regularization)
                self.set_regularization_weights(**regularization_weights)
                ru['olnmpc',i]=self.solve_olrnmpc(**olrnmpc_args) if self.linapprox else self.solve_olnmpc()# solves the olnmpc problem
                self.create_suffixes_nmpc()
                ru['sens',i]=self.sens_k_aug_nmpc()
                #updating phase
                self.create_measurement(meas_noise,**cov_matrices) 
                ru['lsmhe',i]=self.solve_mhe(fix_noise=fix_noise) # solves the mhe problem
                self.update_state_mhe()
                self.compute_offset_state(src_kind="estimated")
                ru['SBU',i]=self.sens_dot_nmpc() 
                ru['forward_simulation',i]=self.forward_simulation()
                self.cycle_iterations()
                if  self.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
                    self.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
                    self.plant_trajectory[i,'solstat'] != ['ok','optimal']:
                    print('ERROR: optimization problem stalled')
                    break
        else:
            for i in range(1,self.nfe_t_0):
                ru['plant_simulation',i]=self.plant_simulation(**disturbance_src)
                # preparation phase
                self.cycle_mhe() 
                self.cycle_nmpc()
                # updating phase
                self.create_measurement(meas_noise,**cov_matrices)  
                ru['lsmhe',i]=self.solve_mhe(fix_noise=fix_noise) # solves the mhe problem
                if self.update_uncertainty_set:
                    self.compute_confidence_ellipsoid()
                self.cycle_ics_mhe(nmpc_as=advanced_step,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc
                self.load_reference_trajectories() if self.obj_type == 'tracking' else None # loads the reference trajectory in olnmpc problem (for regularization)
                self.set_regularization_weights(**regularization_weights)
                ru['olnmpc',i]=self.solve_olrnmpc(**olrnmpc_args) if self.linapprox else self.solve_olnmpc()
                self.cycle_iterations()
                if  self.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
                    self.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
                    self.plant_trajectory[i,'solstat'] != ['ok','optimal']:
                    print('ERROR: optimization problem stalled')
                    break
        i+=1
        ru['plant_simulation',i]=self.plant_simulation(**disturbance_src)
        for l in range(1,i):
            try:
                print('iteration: %i' % l)
                print('open-loop optimal control: ', end='')
                print(self.nmpc_trajectory[l,'solstat'],self.nmpc_trajectory[l,'obj_value'])
                print('constraint inf: ', self.nmpc_trajectory[l,'eps'])
                print('plant: ',end='')
                print(self.plant_trajectory[l,'solstat'])
                print('lsmhe: ', end='')
                print(self.nmpc_trajectory[l,'solstat_mhe'],self.nmpc_trajectory[l,'obj_value_mhe'])
            except:
                pass
        return ru, i
            
            
class msMHEGen(msNMPCGen):
    def __init__(self, **kwargs):
        msNMPCGen.__init__(self, **kwargs)
        self.int_file_mhe_suf = int(time.time())-1

        self.y = kwargs.pop('y', {})

        self.x_noisy = kwargs.pop('x_noisy', [])
        self.noisy_ics = kwargs.pop('noisy_ics', True)
        self.diag_Q_R = kwargs.pop('diag_QR', True)  #: By default use diagonal matrices for Q and R matrices
        self.u = kwargs.pop('u', [])
        self.initial_confidence = kwargs.pop('initial_confidence',1e-6)
        
        self.measurement = {}
        self.noisy_controls = kwargs.pop('noisy_controls', False)
        self.p_noisy = kwargs.pop('p_noisy', {})
        self.process_noise_model = kwargs.pop('process_noise_model',None)
        self.noisy_params = kwargs.pop('noisy_params', False)                
        self.d_mod_mhe = kwargs.pop('d_mod_mhe', None)
        self.update_scenario_tree = kwargs.pop('update_scenario_tree', False)
        self.mhe_confidence_ellipsoids = {}
        
    def create_mhe(self):
        self.lsmhe = self.d_mod_mhe(self.nfe_mhe, self.ncp_t)
        self.lsmhe.name = "lsmhe (Least-Squares MHE)"
        self.lsmhe.create_bounds()
        #self.lsmhe.clear_aux_bounds()
        self.lsmhe.clear_all_bounds()
        self.lsmhe.create_output_relations()
        
        #: Create list of noisy-states vars
        self.xkN_l = []
        self.xkN_nexcl = []
        self.xkN_key = {}
        k = 0
        for x in self.x_noisy:
            n_s = getattr(self.lsmhe, x)  #: Noisy-state
            for jth in self.x[x]:  #: the jth variable
                self.xkN_l.append(n_s[(1, 0) + jth])
                self.xkN_nexcl.append(1)  #: non-exclusion list for active bounds
                self.xkN_key[(x, jth)] = k
                k += 1

        self.lsmhe.xkNk_mhe = Set(initialize=[i for i in range(0, len(self.xkN_l))])  #: Create set of noisy_states
        self.lsmhe.wk_mhe = Var(range(0,self.nfe_mhe), self.lsmhe.xkNk_mhe, initialize=0.0)
        self.lsmhe.Q_mhe = Param(range(0, self.nfe_mhe), self.lsmhe.xkNk_mhe, initialize=1, mutable=True) if self.diag_Q_R\
            else Param(range(0, self.nfe_mhe), self.lsmhe.xkNk_mhe, self.lsmhe.xkNk_mhe,
                             initialize=lambda m, t, i, ii: 1.0 if i == ii else 0.0, mutable=True)  #: Disturbance-weight
        
        # deactivates the continuity constraints in collocation scheme
        for i in self.x_noisy:
            cp_con = getattr(self.lsmhe, "cp_" + i)
            cp_con.deactivate()
        
        #: Create list of measurements vars
        self.yk_l = {}
        self.yk_key = {}
        k = 0
        self.yk_l[1] = []
        for y in self.y:
            m_v = getattr(self.lsmhe, y)  #: Measured "state"
            for jth in self.y[y]:  #: the jth variable
                self.yk_l[1].append(m_v[(1, self.ncp_t) + jth])
                self.yk_key[(y, jth)] = k  #: The key needs to be created only once, that is why the loop was split
                k += 1

        for t in range(2, self.nfe_mhe + 1):
            self.yk_l[t] = []
            for y in self.y:
                m_v = getattr(self.lsmhe, y)  #: Measured "state"
                for jth in self.y[y]:  #: the jth variable
                    self.yk_l[t].append(m_v[(t, self.ncp_t) + jth])

        self.lsmhe.ykk_mhe = Set(initialize=[i for i in range(0, len(self.yk_l[1]))])  #: Create set of measured_vars
        self.lsmhe.nuk_mhe = Var(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=0.0)   #: Measurement noise
        self.lsmhe.yk0_mhe = Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=1.0, mutable=True) #:
    
        # creates constraints that quantify the respective measurement noise
        self.lsmhe.hyk_c_mhe = Constraint(self.lsmhe.fe_t, self.lsmhe.ykk_mhe,
                                          rule=
                                          lambda mod, t, i:mod.yk0_mhe[t, i] - self.yk_l[t][i] - mod.nuk_mhe[t, i] == 0.0)
        self.lsmhe.hyk_c_mhe.deactivate()
        # creates  Identity matrix for weighting of the measurement noise
        self.lsmhe.R_mhe = Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=1.0, mutable=True) if self.diag_Q_R else \
            Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, self.lsmhe.ykk_mhe,
                             initialize=lambda mod, t, i, ii: 1.0 if i == ii else 0.0, mutable=True)
        f = open("file_cv.txt", "w")
        f.close()
        if self.noisy_controls:
            for u in self.u:
                self.lsmhe.add_component("w_" + u + "_mhe", Var(self.lsmhe.fe_t, initialize=0.0))  #: Noise for input vars
                self.lsmhe.add_component("w_" + u + "c_mhe", Constraint(self.lsmhe.fe_t))
                self.lsmhe.equalize_u(direction="r_to_u")
                
                cc = getattr(self.lsmhe, u + "_c")  #: Get the constraint for input
                cc.deactivate()
                
                con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
                wu = getattr(self.lsmhe, "w_" + u + "_mhe")  #: Get the constraint-noisy
                ce = getattr(self.lsmhe, u + "_e")  #: Get the expression
                cp = getattr(self.lsmhe, u)  #: Get the param
    
                con_w.rule = lambda m, i: cp[i] == ce[i] + wu[i]
                con_w.reconstruct()
                con_w.deactivate()

        # initialize as identitiy --> U_mhe = weighting/inv(covariance) of the control noise
        self.lsmhe.U_mhe = Param(self.lsmhe.fe_t, self.u, initialize=1, mutable=True)

        #: Deactivate icc constraints
        if self.noisy_ics:
            # for semi-batch need to introduce also penalty for the initial conditions
            self.lsmhe.noisy_ic = ConstraintList()
            for i in self.x_noisy:
                    ic_con = getattr(self.lsmhe,i + "_icc")
                    ic_exp = getattr(self.lsmhe,i + "_ice")
          
                    # set initial guess
                    xic = getattr(self.lsmhe,i + "_ic")
                    x = getattr(self.lsmhe,i)
                    for j in self.x[i]:
                        k = self.xkN_key[(i,j)] # key that belongs to the certain variable for wk_mhe
                        if j == ():
                            x[(1,0)+j] = xic.value # set reasonable initial guess
                            self.lsmhe.noisy_ic.add(ic_exp == self.lsmhe.wk_mhe[0,k]) # add noisy initial condition
                        else:
                            x[(1,0)+j] = xic[j].value
                            self.lsmhe.noisy_ic.add(ic_exp[j] == self.lsmhe.wk_mhe[0,k]) # add noisy initial condition
                        
                    for k in ic_con.keys():
                        ic_con[k].deactivate() # deactivate the old constraints
        else:
            pass
        
        #: Put the noise in the continuation equations (finite-element)
        j = 0
        self.lsmhe.noisy_cont = ConstraintList()
        for x in self.x_noisy:
            cp_exp = getattr(self.lsmhe, "noisy_" + x)
            for key in self.x[x]:  #: This should keep the same order
                for t in range(1, self.nfe_mhe):
                    self.lsmhe.noisy_cont.add(cp_exp[t, key] == self.lsmhe.wk_mhe[t, j])
                j += 1
         
        self.lsmhe.noisy_cont.deactivate()
       
        self.lsmhe.Q_e_mhe = Expression(
            expr= 1.0/2.0 * sum(
                sum(
                    self.lsmhe.Q_mhe[i, k] * self.lsmhe.wk_mhe[i, k]**2 for k in self.lsmhe.xkNk_mhe)
                for i in range(0, self.nfe_mhe))) if self.diag_Q_R else Expression(
            expr=sum(sum(self.lsmhe.wk_mhe[i, j] *
                         sum(self.lsmhe.Q_mhe[i, j, k] * self.lsmhe.wk_mhe[i, k] for k in self.lsmhe.xkNk_mhe)
                         for j in self.lsmhe.xkNk_mhe) for i in range(0, self.nfe_mhe)))

        self.lsmhe.R_e_mhe = Expression(
            expr=1.0/2.0 * sum(
                sum(
                    self.lsmhe.R_mhe[i, k] * self.lsmhe.nuk_mhe[i, k]**2 for k in self.lsmhe.ykk_mhe)
                for i in self.lsmhe.fe_t)) if self.diag_Q_R else Expression(
            expr=sum(sum(self.lsmhe.nuk_mhe[i, j] *
                         sum(self.lsmhe.R_mhe[i, j, k] * self.lsmhe.nuk_mhe[i, k] for k in self.lsmhe.ykk_mhe)
                         for j in self.lsmhe.ykk_mhe) for i in self.lsmhe.fe_t))
        
        # process_noise_model == time-variant paramters
        self.lsmhe.P_e_mhe = Expression(expr= 0.0)
        if self.process_noise_model == 'params':
            self.pkN_l = []
            self.pkN_nexcl = []
            self.pkN_key = {}
            k = 0
            for p in self.p_noisy:
                try:# check whether parameter is time-variant
                    par = getattr(self.lsmhe, 'p_' + p)
                except:# catch time-invariant parameters
                    self.lsmhe.par_to_var()
                    par = getattr(self.lsmhe, p)
                    par.unfix()
                    print('Parameter ' + p + ' is not time-variant')
                    continue
                for jth in self.p_noisy[p]:  #: the jth variable
                    self.pkN_l.append(par[(1,) + jth])
                    self.pkN_nexcl.append(1)  #: non-exclusion list for active bounds
                    self.pkN_key[(p, jth)] = k
                    k += 1
    
            self.lsmhe.pkNk_mhe = Set(initialize=[i for i in range(0, len(self.pkN_l))])  #: Create set of noisy_states
            self.lsmhe.dk_mhe = Var(self.lsmhe.fe_t, self.lsmhe.pkNk_mhe, initialize=0.0, bounds=(-0.99,1.0))
            self.lsmhe.P_mhe = Param(self.lsmhe.pkNk_mhe, initialize=1.0, mutable=True)
            self.lsmhe.noisy_pars = ConstraintList()
            
            for p in self.p_noisy:
                try:# check whether parameter is time-variant
                    par = getattr(self.lsmhe, 'p_' + p)
                except:# catch time-invariant parameters
                    continue
                for key in self.p_noisy[p]:
                    j = self.pkN_key[(p,key)]
                    for t in range(1,self.nfe_mhe + 1):
                        self.lsmhe.noisy_pars.add(par[(t,)+key] - 1.0 - self.lsmhe.dk_mhe[t,j] == 0.0)
                        par[(t,)+key].unfix()
 
                    
            self.lsmhe.P_e_mhe.expr = 1.0/2.0 * sum(sum(self.lsmhe.P_mhe[k] * self.lsmhe.dk_mhe[i, k]**2 \
                                                        for k in self.lsmhe.pkNk_mhe) \
                                                    for i in self.lsmhe.fe_t)
        elif self.process_noise_model == 'params_bias':
            self.pkN_l = []
            self.pkN_nexcl = []
            self.pkN_key = {}
            k = 0
            for p in self.p_noisy:
                try:# check whether parameter is time-variant
                    par = getattr(self.lsmhe, 'p_' + p)
                except:# catch time-invariant parameters
                    print('Parameter ' + p + ' is not time-variant')
                    continue
                for jth in self.p_noisy[p]:  #: the jth variable
                    self.pkN_l.append(par[(1,) + jth])
                    self.pkN_nexcl.append(1)  #: non-exclusion list for active bounds
                    self.pkN_key[(p, jth)] = k
                    k += 1
    
            self.lsmhe.pkNk_mhe = Set(initialize=[i for i in range(0, len(self.pkN_l))])  #: Create set of noisy_states
            self.lsmhe.dk_mhe = Var(self.lsmhe.fe_t, self.lsmhe.pkNk_mhe, initialize=0.0, bounds=(-0.5,0.5))
            self.lsmhe.P_mhe = Param(self.lsmhe.pkNk_mhe, initialize=1.0, mutable=True)
            self.lsmhe.noisy_pars = ConstraintList()
            
            for p in self.p_noisy:
                try:# check whether parameter is time-variant
                    par = getattr(self.lsmhe, 'p_' + p)
                except:# catch time-invariant parameters
                    continue
                for key in self.p_noisy[p]:
                    j = self.pkN_key[(p,key)]
                    for t in range(1,self.nfe_mhe + 1):
                        jth = (t,)+key
                        self.lsmhe.noisy_pars.add(par[jth] - 1.0 - self.lsmhe.dk_mhe[t,j] == 0.0)
                        par[jth].unfix()

            self.lsmhe.P_e_mhe.expr = 1.0/2.0 * (sum(self.lsmhe.P_mhe[k] * self.initial_confidence * (self.lsmhe.dk_mhe[1, k]-0.0)**2.0 for k in self.lsmhe.pkNk_mhe) +\
                                                sum(sum(self.lsmhe.P_mhe[k] * (self.lsmhe.dk_mhe[i, k]-self.lsmhe.dk_mhe[i-1,k])**2 \
                                                for k in self.lsmhe.pkNk_mhe) for i in range(2,self.nfe_mhe+1)))
            
        else:
            pass # no process noise
            
        expr_u_obf = 0
        
        if self.noisy_controls:
            for i in self.lsmhe.fe_t:
                for u in self.u:
                    var_w = getattr(self.lsmhe, "w_" + u + "_mhe")  #: Get the constraint-noisy
                    expr_u_obf += self.lsmhe.U_mhe[i, u] * var_w[i] ** 2

        self.lsmhe.U_e_mhe = Expression(expr= 1.0/2.0 * expr_u_obf)  # how about this

        self.lsmhe.Arrival_e_mhe = Expression(expr = 0.0)

        self.lsmhe.obfun_dum_mhe_deb = Objective(sense=minimize,
                                             expr=1.0)
        self.lsmhe.obfun_dum_mhe_deb.deactivate()
        
        self.lsmhe.obfun_dum_mhe = Objective(sense=minimize,
                                             expr=self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe + self.lsmhe.P_e_mhe)
        self.lsmhe.obfun_dum_mhe.deactivate()

        self.lsmhe.obfun_mhe = Objective(sense=minimize,
                                         expr=self.lsmhe.Arrival_e_mhe + self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe + self.lsmhe.P_e_mhe)
        self.lsmhe.obfun_mhe.deactivate()


        # deactivate endpoint constraints
        self.lsmhe.deactivate_epc()
        self.lsmhe.deactivate_pc()

        self._PI = {}  #: Container for the reduced Hessian
        self.xreal_W = {}
        self.curr_m_noise = {}   #: Current measurement noise
        self.curr_y_offset = {}  #: Current offset of measurement
        for y in self.y:
            for j in self.y[y]:
                self.curr_m_noise[(y, j)] = 0.0
                self.curr_y_offset[(y, j)] = 0.0

        self.s_estimate = {}
        self.s_real = {}
        for x in self.x_noisy:
            self.s_estimate[x] = []
            self.s_real[x] = []

        self.y_estimate = {}
        self.y_real = {}
        self.y_noise_jrnl = {}
        self.yk0_jrnl = {}
        for y in self.y:
            self.y_estimate[y] = []
            self.y_real[y] = []
            self.y_noise_jrnl[y] = []
            self.yk0_jrnl[y] = []        
    
                
    def cycle_mhe(self,**kwargs):
        m = self.lsmhe if self._c_it > 1 else self.recipe_optimization_model
        initialguess = kwargs.pop('init',self.store_results(m))
        self.create_mhe()
        # open the parameters as degree of freedom
        if self.noisy_params: 
            self.lsmhe.par_to_var()
            for p in self.p_noisy:
                p_mhe = getattr(self.lsmhe,p)
                for key in self.p_noisy[p]:
                    pkey = None if key == () else key
                    p_mhe[pkey].unfix()
        
        # load initialguess from previous iteration lsmhe and olnmpc for the missing element
        if self._c_it == 1:
            for var in self.lsmhe.component_objects(Var):
                for key in var.index_set():
                    try:
                        var[key].value = initialguess[(var.name,key)]
                    except KeyError:
                        try:
                            if type(key) == int:
                                aux_key = (key,) + (1,)
                            elif key == None:
                                continue
                            else:
                                aux_key = (1,) + key[1:] + (1,)
                            var[key].value = initialguess[(var.name,aux_key)] # xxx
                        except KeyError: # fallback strategy
                            continue
                        except AttributeError: # fallback strategy
                            continue
        else:
            for var in self.lsmhe.component_objects(Var):
                for key in var.index_set():
                    try:
                        var[key].value = initialguess[(var.name,key)]
                    except KeyError:
                        try:
                            var_nmpc = getattr(self.olnmpc, var.name)
                            if type(key) == int:
                                aux_key = (key,) + (1,)
                            elif key == None:
                                continue
                            else:
                                aux_key = (1,) + key[1:] + (1,)
                            var[key].value = var_nmpc[aux_key].value # xxx
                        except KeyError: # fallback strategy
                            continue
                        except AttributeError: # fallback strategy
                            continue
                
            
        # adjust the time intervals via model parameter self.lsmhe.fe_dist[i]
        self.lsmhe.tf.fix(self.recipe_optimization_model.tf[1,1].value) # base is set via recipe_optimization_model
        for i in self.lsmhe.fe_t:
            self.lsmhe.fe_dist[i] = (self.nmpc_trajectory[i,'tf']-self.nmpc_trajectory[i-1,'tf'])/self.lsmhe.tf.value
        
        # fix the applied controls in the model:    
        # in case EVM is used --> slacks are added and penalized
        for u in self.u:
            control = getattr(self.lsmhe, u)
            control[self.nfe_mhe].value = self.curr_u[u]
            control.fix()
        
        self.lsmhe.equalize_u()
        
        # activate whats necessary + leggo:
        self.lsmhe.obfun_mhe.activate() # objective function!
        self.lsmhe.noisy_cont.activate() # noisy constraints!
        self.lsmhe.hyk_c_mhe.activate()
        self.lsmhe.eobj.deactivate()
        if self.noisy_controls:
            for u in self.u:
                con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
                con_w.activate()

    def set_measurement_prediction(self,y_cov = {},q_cov = {},u_cov = {},p_cov = {},**kwargs):
        # needs to be called for construction of lsmhe
        # however only of importance when asMHE is applied
        results = kwargs.pop('prediction',self.store_results(self.forward_simulation_model))
        measured_state = {}
        for x in self.x:
            for j in self.x[x]:
                measured_state[(x,j)] = results[(x,(1,3)+j+(1,))]
        for y in self.y:
            for j in self.y[y]:
                measured_state[(y,j)] = results[(y,(1,3)+j+(1,))]
        self.measurement[self._c_it] = deepcopy(measured_state)
 
        # reapply the measured variables
        for t in self.lsmhe.fe_t:
            for y in self.y:
                for j in self.y[y]:
                    vni = self.yk_key[(y,j)]
                    self.lsmhe.yk0_mhe[t,vni] = self.measurement[t][(y,j)]
                    
        # redo the initialization of estimates and matrices
        self.set_covariance_y(y_cov)
        self.set_covariance_w(q_cov)
        if self.noisy_controls:
            self.set_covariance_u(u_cov)
        if self.process_noise_model == 'params' \
           or self.process_noise_model == 'params_bias':
            self.set_covariance_p(p_cov)
                            
    def create_measurement(self,var_dict,y_cov = {},q_cov = {},u_cov = {},p_cov = {}):
        """ Args:
                var_dict = {(x.name,(add_indices)):relative variance for measurement noise}
                results = {(x.name,(add_indices)):results usually obtained by plant simulation}
        """
        # creates measurement for all variables that are in var_dict        
        # prediction for these variables must have been generated beforehand 
        
        # Sets possibly more measured states than are loaded into the state estimation problem
        # what is specified in self.y determines which variables are considered as measured for estimation
        measured_state = {}
        for key in var_dict:
                yvar = getattr(self.plant_simulation_model, key[0])
                j = key[1] + (1,)
                sigma = var_dict[key]
                rand = np.random.normal(loc=0.0, scale=sigma)
                if abs(rand) > 2*sigma:
                    rand = -2*sigma if rand < 0.0 else 2*sigma
                else:
                    pass
                measured_state[key] = (1-rand)*yvar[(1,3)+j].value
        self.measurement[self._c_it] = deepcopy(measured_state)
        
        # reapply the measured variables
        for t in self.lsmhe.fe_t:
            for y in self.y:
                for j in self.y[y]:
                    vni = self.yk_key[(y,j)]
                    self.lsmhe.yk0_mhe[t,vni] = self.measurement[t][(y,j)]
                    
        # redo the initialization of estimates and matrices
        self.set_covariance_y(y_cov)
        self.set_covariance_w(q_cov)
        if self.noisy_controls:
            self.set_covariance_u(u_cov)
        if self.process_noise_model == 'params' \
           or self.process_noise_model == 'params_bias':
            self.set_covariance_p(p_cov)
                
    def solve_mhe(self,fix_noise=False):
        self.journalizer('U', self._c_it, 'state estimation problem','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        # fix process noise as degrees of freedom
        if fix_noise:
            self.lsmhe.wk_mhe.fix()
        
        initialguess = self.store_results(self.lsmhe)
        #for var in self.lsmhe.component_objects(Var):
        #        for key in var.index_set():
        #            var[key].value = initialguess[var.name,key]   
                    
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "no"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 500
        with open("ipopt.opt", "w") as f:
            f.write("print_info_string yes")
            f.close()


        result = ip.solve(self.lsmhe, tee=True)
        
        if [str(result.solver.status),str(result.solver.termination_condition)] != ['ok','optimal']:
            for p in self.p_noisy:
                par_mhe = getattr(self.lsmhe, p)
                par_true = getattr(self.plant_simulation_model, p)
                for key in self.p_noisy[p]:
                    pkey = None if key ==() else key
                    print('estimated')
                    par_mhe.pprint()
                    print('true')
                    par_true.pprint()
            for var in self.lsmhe.component_objects(Var):
                for key in var.index_set():
                    var[key].value = initialguess[var.name,key]        
            self.lsmhe.create_bounds()
            self.lsmhe.clear_aux_bounds()
            self.lsmhe.par_to_var()
            result = ip.solve(self.lsmhe, tee=True)
        
        # saves the results to initialize the upcoming mhe problem
        # is up to the user to do it by hand
        output = self.store_results(self.lsmhe) # saves the results to initialize the upcoming mhe problem
    
        if self.adapt_params:
            # {timestep after which the parameters were identified, 'e_pars':dictionary with estimates}
            for p in self.p_noisy:
                p_mhe = getattr(self.lsmhe,p)
                for key in self.p_noisy[p]:
                    pkey = key if key != () else None 
                    self.curr_epars[(p,key)] = p_mhe[pkey].value
                    
            self.nmpc_trajectory[self._c_it,'e_pars'] = deepcopy(self.curr_epars)
            
        # saves the predicted initial_values for the states
        for x in self.x:
            for j in self.x[x]:
                self.initial_values[(x,j)] = output[(x,(self.nfe_mhe,3)+j)]
                
        # load solution status in self.nmpc_trajectory
        self.nmpc_trajectory[self._c_it,'solstat_mhe'] = [str(result.solver.status),str(result.solver.termination_condition)]
        self.nmpc_trajectory[self._c_it,'obj_value_mhe'] = value(self.lsmhe.obfun_mhe)
        
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end) 
        
    def cycle_ics_mhe(self, nmpc_as = False, mhe_as = False, alpha=0.95):
        if self.noisy_params and self._c_it > 1:
            ##################################################################
            # comute principle components of approximate 95%-confidence region
            ##################################################################
            try: # adapt parameters iff estimates are confident enough
                dimension = int(np.round(np.sqrt(len(self.mhe_confidence_ellipsoids[1]))))
                confidence = chi2.isf(1-alpha,dimension)
                A_dict = self.mhe_confidence_ellipsoids[self._c_it-1]
                # assemble dimension x dimension array
                rows = {}
                for m in range(dimension):
                        rows[m] = np.array([A_dict[(m,i)] for i in range(dimension)])
                A = 1/confidence*np.array([np.array(rows[i]) for i in range(dimension)]) # shape matrix of confidence ellipsoid, reduced hessian in this case
                U, s, V = np.linalg.svd(A) # singular value decomposition of shape matrix 
                radii = 1/np.sqrt(s) # radii --
                
                if self.adapt_params:
                    for p in self.p_noisy:
                        p_nom = getattr(self.olnmpc,p)
                        p_mhe = getattr(self.lsmhe,p)
                        for key in self.p_noisy[p]:
                            pkey = key if key != () else None
                            index = self.PI_indices[p,key]
                            dev = -1e8
                            for m in range(dimension):
                                 dev = max(dev,(abs(radii[m]*U[index][m]) + p_mhe[pkey].value)/p_mhe[pkey].value)
                            if dev < 1 + self.estimate_acceptance:
                                self.curr_epars[(p,key)] = p_mhe[pkey].value
                                p_nom[pkey].value = p_mhe[pkey].value
                            else:
                                self.curr_epars[(p,key)] = p_nom[pkey].value
                            
                    
                                
            except KeyError: # adapt parameters blindly
                if self.adapt_params:
                    for p in self.p_noisy:
                        p_nom = getattr(self.olnmpc)
                        p_mhe = getattr(self.lsmhe,p)
                        for key in self.p_noisy[p]:
                            pkey = key if key != () else None 
                            self.curr_epars[(p,key)] = p_mhe[pkey].value
                            p_nom[pkey].value = p_mhe[pkey].value
                    else:
                        pass

            ###############################################################
            ###############################################################
            # alternative: use the projections on the axis
            # projection x^T A x = 1
            # A_k = A with kth row and kth column with zeros and akk = 1
            # solve A_k*D_k = Z_k 
            # with Z_k = -a_k (kth column of A) and Z_k,k = 1.0
            # delta_x_k = np.sqrt(a_k^T*D_k)
            
                #
            if self.update_scenario_tree:
                dev = {(p,key):self.confidence_threshold for p in self.p_noisy for key in self.p_noisy[p]}
                for p in self.p_noisy:
                    p_mhe = getattr(self.lsmhe,p)
                    for key in self.p_noisy[p]:
                        pkey = key if key != () else None
                        k = self.PI_indices[p,key]
                        A_k = deepcopy(A)
                        A_k[:,k] = 0.0
                        A_k[k,:] = 0.0
                        A_k[k,k] = 1.0
                        Z_k = np.zeros(dimension)
                        Z_k[:] = -A[:,k]
                        Z_k[k] = 1.0
                        a_k = A[k,:]
                        D_k = np.linalg.solve(A_k,Z_k)
                        dev_k = np.sqrt(1/np.dot(a_k,D_k))/p_mhe[pkey].value
                        # use new interval if robustness_threshold < dev_k < confidence_threshold
                        dev[(p,key)] = min(self.confidence_threshold,max(self.robustness_threshold, dev_k*p_mhe[pkey].value/self.nominal_parameter_values[(p,key)])) \
                                        * self.nominal_parameter_values[(p,key)]/p_mhe[pkey].value
                                                
                # update scenario_tree
                for k in self.st:
                    for index in self.st[k][3]:
                        if k[1] != 1: # 1 is nominal scenario
                            try:
                                if self.uncertainty_set == {}:
                                    self.st[k][3][index] =  1.0 + dev[index] if self.st[k][3][index] > 1.0 else 1.0 - dev[index]
                                else:
                                    p_nom = self.nominal_parameter_values[index]
                                    p_mhe = self.curr_epars[index]
                                    lb = min(max((1.0 - dev[index])*p_mhe,(1.0 + self.uncertainty_set[index][0])*p_nom)/p_mhe,1.0-1e-12)
                                    ub = max(min((1.0 + dev[index])*p_mhe,(1.0 + self.uncertainty_set[index][1])*p_nom)/p_mhe,1.0+1e-12)
                                    self.st[k][3][index] = ub if self.st[k][3][index] > 1.0 else lb
                            except:
                                # catch that not every uncertain parameter included in the scenario tree
                                # is necessarily estimated
                                pass
                        else:
                            continue   
                self.olnmpc.set_scenarios()
                self.nmpc_trajectory[self._c_it, 'st'] = deepcopy(self.st)
        
        
        # cycle estimates for initial condition
        if nmpc_as and mhe_as: # both nmpc and mhe use advanced step schme
            ivs = self.initial_values
        elif nmpc_as and not(mhe_as): # only nmpc uses advanced step scheme
            ivs = self.curr_pstate
        else: # nothing uses advanced step scheme/no measurement noise
            ivs = self.initial_values
            
        for x in self.x:
            xic = getattr(self.olnmpc, x+'_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = ivs[(x,j)]

    def set_covariance_y(self, cov_dict):
        """Sets covariance(inverse) for the measurements.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(meas_name, j), (meas_name, k), time]
        Returns:
            None
        """
        # measured states R_mhe
        rtarget = getattr(self.lsmhe, "R_mhe")
        for key in cov_dict if self.diag_Q_R else cov_dict[0]:
            # 1. check whether the variable is even a measured one
            if key[0] in self.yk_key: 
                # 2. if yes: compute the respective weight as follows:
                vni = key[0]
                vnj = key[1]
                v_i = self.yk_key[vni]
                v_j = self.yk_key[vnj]
                for t in range(1,self.nfe_mhe+1):
  #                  if self.diag_Q_R:                
                      rtarget[t, v_i] = 1 / max(abs(cov_dict[vni, vnj]*self.measurement[t][vni]),1e-4)**2
 #                   else:
                        # only allow for diagonal measurement covariance matrices 
#                        rtarget[t, v_i, v_j] = 1 / max(abs(cov_dict[t][vni, vnj]*self.measurement[t][vni]),1e-4)**2
            else:
                continue

    def set_covariance_w(self, cov_dict, set_bounds=True):
        """Sets covariance(inverse) for the noisy states.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(state_name, j), (state_name, k), time]
        Returns:
            None
        """
        # disturbances
        qtarget = getattr(self.lsmhe, "Q_mhe")
        w = getattr(self.lsmhe, 'wk_mhe')
        for key in cov_dict if self.diag_Q_R else cov_dict[0]:
            if key[0] in self.xkN_key:
                vni = key[0]
                vnj = key[1]
                v_i = self.xkN_key[vni]
                v_j = self.xkN_key[vnj]
                xic = getattr(self.lsmhe, vni[0] + "_ic")
                xkey = None if vni[1] == () else vni[1]
                if self.diag_Q_R:
                    qtarget[0, v_i] = 1.0 / (max(abs(cov_dict[vni, vnj]*xic[xkey].value),1e-4))**2 # .00001
                    if set_bounds:
                        if cov_dict[vni, vnj] != 0:
                            confidence = 3*cov_dict[vni, vnj]*xic[xkey].value
                            w[0,v_i].setlb(-confidence)
                            w[0,v_i].setub(confidence)
                            if abs(w[0,v_i].lb - w[0,v_i].ub) < 1e-3:
                                w[0,v_i].fix(0.0)
                        else:
                            w[0,v_i].fix(0.0)
                else:
                    qtarget[0, v_i, v_j] = cov_dict[0][vni, vnj]
                    # no bounding available for this case, do by hand
                    
                for t in range(1,self.nfe_mhe):
                    if self.diag_Q_R:
                        qtarget[t, v_i] = 1 / (max(abs(cov_dict[vni, vnj]*self.nmpc_trajectory[t,vni]),1e-4))**2 # 0.00001
                        if set_bounds:
                            if cov_dict[vni, vnj] != 0.0:
                                confidence = 10*abs(cov_dict[vni,vnj]*self.nmpc_trajectory[t,vni])
                            else:
                                confidence = 0.0 
                            w[t,v_i].setlb(-confidence)
                            w[t,v_i].setub(confidence)
                            if abs(w[t,v_i].lb - w[t,v_i].ub) < 1e-3:
                                w[t,v_i].fix()

                    else:
                        if (vni,vnj) in cov_dict[t]:
                            qtarget[t, v_i, v_j] = cov_dict[t][vni, vnj]
                        else:
                            print('no weight for element : \n',vni,vnj,'\n specified')
                            qtarget[t, v_i, v_j] = 1e8

            else:
                print(key[0], ' is not a noisy state variable')

    def set_covariance_u(self, cov_dict):
        """Sets covariance(inverse) for the states.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(state_name, j), (state_name, k), time]
        Returns:
            None
        """
        utarget = getattr(self.lsmhe, "U_mhe")
        for key in cov_dict:
            vni = key[0]
            for _t in range(1,self.nfe_mhe+1):
                aux_key = self.nfe_t_0 if _t > self.nfe_t_0 else (_t,1) # tailored to my code basically
                utarget[_t, vni] = 1 / (max(abs(cov_dict[key]*self.reference_control_profile[vni,aux_key]),1e-4))**2

    
    def set_covariance_p(self, cov_dict, set_bounds=True):
        ptarget = getattr(self.lsmhe, "P_mhe")
        for key in cov_dict:
            # only
            p = key[0][0]
            index = key[0][1]
            k = self.pkN_key[(p,index)]
            ptarget[k] = 1/cov_dict[key]**2.0

    def compute_measurement_offset(self):
        mhe_y = getattr(self.lsmhe, "yk0_mhe")
        for y in self.y:
            for j in self.y[y]:
                k = self.yk_key[(y, j)]
                mhe_yval = value(mhe_y[self.nfe_mhe, k])
                self.curr_y_offset[(y, j)] = mhe_yval - self.measurement[self._c_it][(y,j)]           
    
    def compute_confidence_ellipsoid(self):
        """ computes confidence ellipsoids for estimated parameters via 1st order approximation"""
        self.journalizer('U',self._c_it,'confidence ellipsoid computation','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name

        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        # set parameters
        for p in self.p_noisy:
            par = getattr(self.lsmhe,p) 
            for j in self.p_noisy[p]:
                if j == ():
                    par.set_suffix_value(self.lsmhe.dof_v, 1)
                else:
                    par[j].set_suffix_value(self.lsmhe.dof_v, 1)
        
        self.lsmhe.ipopt_zL_in.update(self.lsmhe.ipopt_zL_out)
        self.lsmhe.ipopt_zU_in.update(self.lsmhe.ipopt_zU_out)          

        self.journalizer("I", self._c_it, "load_covariance_prior", "K_AUG w red_hess")
        self.k_aug.options["compute_inv"] = ""
        self.k_aug.options["no_barrier"] = ""
        self.k_aug.options["no_scale"] = ""
        
        try:
            self.k_aug.solve(self.lsmhe, tee=True)
        except:
            self.nmpc_trajectory[self._c_it,'solstat_mhe'] = ['Inversion of Reduced Hessian failed','Inversion of Reduced Hessian failed']
      
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        self._PI.clear()
        self.PI_indices = {}
    
        for p in self.p_noisy:
            par = getattr(self.lsmhe,p) 
            for j in self.p_noisy[p]:
                if j == ():
                    aux = par.get_suffix_value(self.lsmhe.rh_name) 
                    if aux == None:
                        self.PI_indices[par.name,j] = 0
                    else:
                        self.PI_indices[par.name,j] = aux
                else:
                    aux = par[j].get_suffix_value(self.lsmhe.rh_name)
                    if aux == None:
                        self.PI_indices[par.name,j] = 0
                    else:
                        self.PI_indices[par.name,j] = aux
                   
        #"inv_.in" gives the reduced hessian which is the shape matrix in x^T A x = 1
        # inv_.in contains the reduced hessian
        # read from file and store in _PI according to order specified in _PI
        with open("inv_.in", "r") as rh:
            ll = []
            l = rh.readlines()
            row = 0
            for i in l:
                ll = i.split()
                col = 0
                for j in ll:
                    self._PI[row, col] = float(j)
                    col += 1
                row += 1
            rh.close()
        print("-" * 120)
        print("I[[load covariance]] e-states nrows {:d} ncols {:d}".format(len(l), len(ll)))
        print("-" * 120)
        
        # unscaled matrix
        self.mhe_confidence_ellipsoids[self._c_it] = deepcopy(self._PI)
        
        # scaled shape matrix of confidence ellipsoid based on the current estimate
        dim = len(self.PI_indices)
        S = np.zeros((dim,dim))
        confidence = chi2.isf(1.0-0.95,dim)
        rows = {}
        m = 0
        for p in self.PI_indices:
            p_mhe = getattr(self.lsmhe,p[0])
            key = p[1] if p[1] != () else None
            S[self.PI_indices[p]][self.PI_indices[p]] = p_mhe[key].value
            rows[m] = np.array([self._PI[(m,i)] for i in range(dim)])
            m += 1
        A = 1/confidence*np.array([np.array(rows[i]) for i in range(dim)]) 
        self._scaled_shape_matrix = np.dot(S.transpose(),np.dot(A,S)) # scaled shape matrix = scaled reduced hessian
        
        
        # alternatively much more efficient to just compute inverse of reduced hessian and scale that.
        # not is not required to compute the actual shape matrix of the ellipsoid
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
        
    """ for advanced-step MHE """
    def create_rh_sfx(self, set_suffix=True):
        """Creates relevant suffixes for k_aug (prior at fe=2) (Reduced_Hess)
        Args:
            set_suffix (bool): True if update must be done
        Returns:
            None
        """
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name

        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)

        if set_suffix:
            for key in self.x_noisy:
                var = getattr(self.lsmhe, key)
                for j in self.x[key]:
                    var[(2, 0) + j].set_suffix_value(self.lsmhe.dof_v, 1)

    def create_sens_suffix_mhe(self, set_suffix=True):
        """Creates relevant suffixes for k_aug (Sensitivity)
        Args:
            set_suffix (bool): True if update must be done
        Returns:
            None"""
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name
        if set_suffix:
            for key in self.x_noisy:
                var = getattr(self.lsmhe, key)
                for j in self.x[key]:
                    var[(self.nfe_mhe, self.ncp_t) + j].set_suffix_value(self.lsmhe.dof_v, 1)

    def compute_y_offset(self, noisy=True):
        mhe_y = getattr(self.lsmhe, "yk0_mhe")
        for y in self.y:
            plant_y = getattr(self.d1, y)
            for j in self.y[y]:
                k = self.yk_key[(y, j)]
                mhe_yval = value(mhe_y[self.nfe_mhe, k])
                plant_yval = value(plant_y[(1, self.ncp_t) + j])
                y_noise = self.curr_m_noise[(y, j)] if noisy else 0.0
                self.curr_y_offset[(y, j)] = mhe_yval - plant_yval - y_noise

    def sens_dot_mhe(self):
        """Updates suffixes, solves using the dot_driver"""
        self.journalizer("I", self._c_it, "sens_dot_mhe", "Set-up")

        if hasattr(self.lsmhe, "npdp"):
            self.lsmhe.npdp.clear()
        else:
            self.lsmhe.npdp = Suffix(direction=Suffix.EXPORT)
        self.create_sens_suffix_mhe()
        for y in self.y:
            for j in self.y[y]:
                k = self.yk_key[(y, j)]
                self.lsmhe.hyk_c_mhe[self.nfe_mhe, k].set_suffix_value(self.lsmhe.npdp, self.curr_y_offset[(y, j)])

        # with open("somefile0.txt", "w") as f:
        #     self.lsmhe.x.display(ostream=f)
        #     self.lsmhe.M.display(ostream=f)
        #     f.close()
        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        #: Looks for the file with the timestamp
        self.lsmhe.set_suffix_value(self.lsmhe.f_timestamp, self.int_file_mhe_suf)

        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        self.journalizer("I", self._c_it, "sens_dot_mhe", self.lsmhe.name)

        results = self.dot_driver.solve(self.lsmhe, tee=True, symbolic_solver_labels=True)
        self.lsmhe.solutions.load_from(results)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        # with open("somefile1.txt", "w") as f:
        #     self.lsmhe.x.display(ostream=f)
        #     self.lsmhe.M.display(ostream=f)
        #     f.close()

        ftiming = open("timings_dot_driver.txt", "r")
        s = ftiming.readline()
        ftiming.close()
        k = s.split()
        self._dot_timing = k[0]

    def sens_k_aug_mhe(self):
        self.journalizer("I", self._c_it, "sens_k_aug_mhe", "k_aug sensitivity")
        self.lsmhe.ipopt_zL_in.update(self.lsmhe.ipopt_zL_out)
        self.lsmhe.ipopt_zU_in.update(self.lsmhe.ipopt_zU_out)
        self.journalizer("I", self._c_it, "sens_k_aug_mhe", self.lsmhe.name)

        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        #: Now, the sensitivity step will have the timestamp for dot_in

        self.lsmhe.set_suffix_value(self.lsmhe.f_timestamp, self.int_file_mhe_suf)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)
        self.create_sens_suffix_mhe()
        results = self.k_aug_sens.solve(self.lsmhe, tee=True, symbolic_solver_labels=True)
        self.lsmhe.solutions.load_from(results)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)
        ftimings = open("timings_k_aug.txt", "r")
        s = ftimings.readline()
        ftimings.close()
        self._k_timing = s.split()

    def update_state_mhe(self, as_nmpc_mhe_strategy=False):
        # Improvised strategy
        if as_nmpc_mhe_strategy:
            self.journalizer("I", self._c_it, "update_state_mhe", "offset ready for asnmpcmhe")
            for x in self.x:
                xvar = getattr(self.lsmhe, x)
                xic = getattr(self.olnmpc, x + "_ic")
                for j in self.x[x]:
                    xkey = None if j == () else j
                    self.curr_state_offset[(x, j)] = value(xic[xkey])- value(xvar[self.nfe_mhe, self.ncp_t, j])
                    #print("state !", self.curr_state_offset[(x, j)])

        for x in self.x:
            xvar = getattr(self.lsmhe, x)
            for j in self.x[x]:
                self.curr_estate[(x, j)] = value(xvar[self.nfe_mhe, self.ncp_t, j])
                
    def run(self,meas_noise={},cov_matrices={},regularization_weights={},disturbance_src={},\
            stgen_args = {}, stgen=False,advanced_step=False,fix_noise=False):
        ru = {}
        # off-line open-loop control
        ru['recipe_optimization']=self.recipe_optimization()
        
        self.set_reference_state_trajectory(self.get_state_trajectory(self.recipe_optimization_model))
        self.set_reference_control_profile(self.get_control_profile(self.recipe_optimization_model))
        self.create_nmpc() if self.obj_type == 'tracking' else self.create_enmpc() 
        self.load_reference_trajectories() if self.obj_type == 'tracking' else None
        self.create_mhe()
        # on-line control
        if advanced_step:
            for i in range(1,self.nfe_t_0):
                ru['plant_simulation',i]=self.plant_simulation(**disturbance_src)
                # preparation phase
                self.cycle_mhe() 
                if stgen:
                    ru['stgen',i]=self.scenario_tree_generation(**stgen_args)
                self.cycle_nmpc() 
                if self.update_scenario_tree:
                    ru['confidence_ellipsoid_computation',i]=self.compute_confidence_ellipsoid()
                self.cycle_ics_mhe(nmpc_as=advanced_step,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc
                if self.obj_type == 'tracking':
                    self.load_reference_trajectories()
                self.set_regularization_weights(**regularization_weights)
                ru['olnmpc',i]=self.solve_olnmpc() # solves the olnmpc problem
                self.create_suffixes_nmpc()
                ru['sens',i]=self.sens_k_aug_nmpc()
                #updating phase
                self.create_measurement(meas_noise,**cov_matrices)
                ru['lsmhe',i]=self.solve_mhe(fix_noise=fix_noise) # solves the mhe problem
                self.update_state_mhe()
                self.compute_offset_state(src_kind="estimated")
                ru['SBU',i]=self.sens_dot_nmpc() 
                ru['forward_simulation',i]=self.forward_simulation()
                self.cycle_iterations()
                if  self.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
                    self.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
                    self.plant_trajectory[i,'solstat'] != ['ok','optimal']:
                    print('ERROR: optimization problem stalled')
                    break
        else:
            for i in range(1,self.nfe_t_0):
                ru['plant_simulation',i]=self.plant_simulation(**disturbance_src)
                # preparation phase
                self.cycle_mhe() 
                if stgen:
                    ru['stgen',i]=self.scenario_tree_generation(**stgen_args)
                self.cycle_nmpc()
                # updating phase
                self.create_measurement(meas_noise,**cov_matrices) 
                ru['lsmhe',i]=self.solve_mhe(fix_noise=fix_noise) # solves the mhe problem
                if self.update_scenario_tree:
                    ru['confidence_ellipsoid_computation',i]=self.compute_confidence_ellipsoid()
                self.cycle_ics_mhe(nmpc_as=advanced_step,mhe_as=False) # writes the obtained initial conditions from mhe into olnmpc
                if self.obj_type == 'tracking':
                    self.load_reference_trajectories() 
                self.set_regularization_weights(**regularization_weights)
                ru['olnmpc',i]=self.solve_olnmpc() # solves the olnmpc problem
                
                self.cycle_iterations()
                if  self.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
                    self.nmpc_trajectory[i,'solstat'] != ['ok','optimal'] or \
                    self.plant_trajectory[i,'solstat'] != ['ok','optimal']:
                    print('ERROR: optimization problem stalled')
                    break    
        i+=1
        ru['plant_simulation',i]=self.plant_simulation(**disturbance_src)
        for l in range(1,i):
            try:
                print('iteration: %i' % l)
                print('open-loop optimal control: ', end='')
                print(self.nmpc_trajectory[l,'solstat'],self.nmpc_trajectory[l,'obj_value'])
                print('constraint inf: ', self.nmpc_trajectory[l,'eps'])
                print('plant: ',end='')
                print(self.plant_trajectory[l,'solstat'])
                print('lsmhe: ', end='')
                print(self.nmpc_trajectory[l,'solstat_mhe'],self.nmpc_trajectory[l,'obj_value_mhe'])
            except:
                pass
        return ru, i
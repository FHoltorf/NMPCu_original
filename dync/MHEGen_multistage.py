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
from main.dync.DynGen_adjusted import DynGen
from main.dync.NMPCGen_multistage import NmpcGen
import numpy as np
from itertools import product
import sys, os, time
from copy import deepcopy
from scipy.stats import chi2
from pyomo import *

__author__ = "David M Thierry @dthierry"
"""Not yet. Our people, they don't understand."""

class MheGen(NmpcGen):
    def __init__(self, **kwargs):
        NmpcGen.__init__(self, **kwargs)
        self.int_file_mhe_suf = int(time.time())-1

        # Need a list of relevant measurements 
        self.y = kwargs.pop('y', [])
        self.y_vars = kwargs.pop('y_vars', {})

        # Need a list or relevant noisy-states z

        self.x_noisy = kwargs.pop('x_noisy', [])
        self.x_vars = kwargs.pop('x_vars', {})
        self.deact_ics = kwargs.pop('del_ics', True)
        self.diag_Q_R = kwargs.pop('diag_QR', True)  #: By default use diagonal matrices for Q and R matrices
        self.u = kwargs.pop('u', [])
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
        #self.nfe_t_total = kwargs('nfe_t_total',5)
        self.measurement = {}
        self.noisy_inputs = kwargs.pop('noisy_inputs', False)
        self.p_noisy = kwargs.pop('p_noisy', {})
        self.noisy_params = kwargs.pop('noisy_params', False)                
        self.d_mod_mhe = kwargs.pop('d_mod_mhe', None)
        self.update_scenario_tree = kwargs.pop('update_scenario_tree', False)
        self.mhe_confidence_ellipsoids = {}
        
    def create_mhe(self):
        self.lsmhe = self.d_mod_mhe(self.nfe_mhe, self.ncp_t, _t=self._t)
        self.lsmhe.name = "lsmhe (Least-Squares MHE)"
        self.lsmhe.create_bounds()
        self.lsmhe.create_output_relations()
        #: create x_pi constraint

        #: Create list of noisy-states vars
        self.xkN_l = []
        self.xkN_nexcl = []
        self.xkN_key = {}
        k = 0
        for x in self.x_noisy:
            n_s = getattr(self.lsmhe, x)  #: Noisy-state
            for jth in self.x_vars[x]:  #: the jth variable
                self.xkN_l.append(n_s[(1, 0) + jth])
                self.xkN_nexcl.append(1)  #: non-exclusion list for active bounds
                self.xkN_key[(x, jth)] = k
                k += 1

        self.lsmhe.xkNk_mhe = Set(initialize=[i for i in range(0, len(self.xkN_l))])  #: Create set of noisy_states
        self.lsmhe.wk_mhe = Var(range(0,self.nfe_mhe), self.lsmhe.xkNk_mhe, initialize=0.0)
        
        
        self.lsmhe.Q_mhe = Param(range(0, self.nfe_mhe), self.lsmhe.xkNk_mhe, initialize=1, mutable=True) if self.diag_Q_R\
            else Param(range(0, self.nfe_mhe), self.lsmhe.xkNk_mhe, self.lsmhe.xkNk_mhe,
                             initialize=lambda m, t, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Disturbance-weight
        
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
            for jth in self.y_vars[y]:  #: the jth variable
                self.yk_l[1].append(m_v[(1, self.ncp_t) + jth])
                self.yk_key[(y, jth)] = k  #: The key needs to be created only once, that is why the loop was split
                k += 1

        for t in range(2, self.nfe_mhe + 1):
            self.yk_l[t] = []
            for y in self.y:
                m_v = getattr(self.lsmhe, y)  #: Measured "state"
                for jth in self.y_vars[y]:  #: the jth variable
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
        if self.noisy_inputs:
            for u in self.u:
                # cv = getattr(self.lsmhe, u)  #: Get the param
                # c_val = [value(cv[i]) for i in cv.keys()]  #: Current value
                # self.lsmhe.del_component(cv)  #: Delete the param
                # self.lsmhe.add_component(u + "_mhe", Var(self.lsmhe.fe_t, initialize=lambda m, i: c_val[i-1]))
                self.lsmhe.add_component("w_" + u + "_mhe", Var(self.lsmhe.fe_t, initialize=0.0))  #: Noise for input vars
                self.lsmhe.add_component("w_" + u + "c_mhe", Constraint(self.lsmhe.fe_t))
                self.lsmhe.equalize_u(direction="r_to_u")
                
                cc = getattr(self.lsmhe, u + "_c")  #: Get the constraint for input
                cc.deactivate()
                
                con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
                var_w = getattr(self.lsmhe, "w_" + u + "_mhe")  #: Get the constraint-noisy
                ce = getattr(self.lsmhe, u + "_e")  #: Get the expression
                cp = getattr(self.lsmhe, u)  #: Get the param
    
                con_w.rule = lambda m, i: cp[i] == ce[i] + var_w[i]
                con_w.reconstruct()
                con_w.deactivate()

        # always Identitiy --> U_mhe is weighting of the control noise
        self.lsmhe.U_mhe = Param(self.lsmhe.fe_t, self.u, initialize=1, mutable=True)

        #: Deactivate icc constraints
        if self.deact_ics:
            pass

        #: Maybe only for a subset of the states
        else:
            # for semi-batch need to introduce also penalty for the initial conditions
            self.lsmhe.noisy_ic = ConstraintList()
            for i in self.x_noisy:
                    ic_con = getattr(self.lsmhe,i + "_icc")
                    ic_exp = getattr(self.lsmhe,i + "_ice")
          
                    # set initial guess
                    xic = getattr(self.lsmhe,i + "_ic")
                    x = getattr(self.lsmhe,i)
                    for j in self.x_vars[i]:
                        k = self.xkN_key[(i,j)] # key that belongs to the certain variable
                        if j == ():
                            x[(1,0)+j] = xic.value # set reasoable initial guess
                            self.lsmhe.noisy_ic.add(ic_exp == 1000*self.lsmhe.wk_mhe[0,k]) # add noisy initial condition
                        else:
                            x[(1,0)+j] = xic[j].value
                            self.lsmhe.noisy_ic.add(ic_exp[j] == 1000*self.lsmhe.wk_mhe[0,k]) # add noisy initial condition
                        
                    for k in ic_con.keys():
                        ic_con[k].deactivate() # deactivate the old constraints
                    
        #: Put the noise in the continuation equations (finite-element)
        j = 0
        self.lsmhe.noisy_cont = ConstraintList()
        for i in self.x_noisy:
            #cp_con = getattr(self.lsmhe, "cp_" + i)
            cp_exp = getattr(self.lsmhe, "noisy_" + i)
            # self.lsmhe.del_component(cp_con)
            for k in self.x_vars[i]:  #: This should keep the same order
                for t in range(1, self.nfe_mhe):
                    # pass
                    self.lsmhe.noisy_cont.add(cp_exp[t, k] == self.lsmhe.wk_mhe[t, j])
                    # self.lsmhe.noisy_cont.add(cp_exp[t, k] == 0.0)
                j += 1
            #cp_con.reconstruct()
        self.lsmhe.noisy_cont.deactivate()

        #: Expressions for the objective function (least-squares)
#        self.lsmhe.Q_e_mhe = Expression(
#            expr=0.5 * sum(
#                sum(
#                    self.lsmhe.Q_mhe[i, k] * self.lsmhe.wk_mhe[i, k]**2 for k in self.lsmhe.xkNk_mhe)
#                for i in range(1, self.nfe_mhe))) if self.diag_Q_R else Expression(
#            expr=sum(sum(self.lsmhe.wk_mhe[i, j] *
#                         sum(self.lsmhe.Q_mhe[i, j, k] * self.lsmhe.wk_mhe[i, k] for k in self.lsmhe.xkNk_mhe)
#                         for j in self.lsmhe.xkNk_mhe) for i in range(1, self.nfe_mmhe)))
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
        expr_u_obf = 0
        
        if self.noisy_inputs:
            for i in self.lsmhe.fe_t:
                for u in self.u:
                    var_w = getattr(self.lsmhe, "w_" + u + "_mhe")  #: Get the constraint-noisy
                    expr_u_obf += self.lsmhe.U_mhe[i, u] * var_w[i] ** 2

        self.lsmhe.U_e_mhe = Expression(expr= 1.0/2.0 * expr_u_obf)  # how about this
#        with open("file_cv.txt", "a") as f:
#            self.lsmhe.U_e_mhe.pprint(ostream=f)
#            f.close()

# DO NOT NEED ARIVAL COST FOR NOW
#        self.lsmhe.Arrival_e_mhe = Expression(
#            expr=0.5 * sum((self.xkN_l[j] - self.lsmhe.x_0_mhe[j]) *
#                     sum(self.lsmhe.PikN_mhe[j, k] * (self.xkN_l[k] - self.lsmhe.x_0_mhe[k]) for k in self.lsmhe.xkNk_mhe)
#                     for j in self.lsmhe.xkNk_mhe))
        self.lsmhe.Arrival_e_mhe = Expression(expr = 0.0)

        self.lsmhe.obfun_dum_mhe_deb = Objective(sense=minimize,
                                             expr=1.0)
        self.lsmhe.obfun_dum_mhe_deb.deactivate()
        
        self.lsmhe.obfun_dum_mhe = Objective(sense=minimize,
                                             expr=self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe)
        self.lsmhe.obfun_dum_mhe.deactivate()

        self.lsmhe.obfun_mhe = Objective(sense=minimize,
                                         expr=self.lsmhe.Arrival_e_mhe + self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe)
        self.lsmhe.obfun_mhe.deactivate()


        # deactivate endpoint constraints
        self.lsmhe.deactivate_epc()
        self.lsmhe.deactivate_pc()

        # with open("file_cv.txt", "a") as f:
        #     self.lsmhe.obfun_mhe.pprint(ostream=f)
        #     f.close()

        self._PI = {}  #: Container of the KKT matrix
        self.xreal_W = {}
        self.curr_m_noise = {}   #: Current measurement noise
        self.curr_y_offset = {}  #: Current offset of measurement
        for y in self.y:
            for j in self.y_vars[y]:
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
            
        # remove all unnecessary bounds to improve reduced hessian computation
        for var in self.lsmhe.component_objects(Var, active=True):
            for key in var.index_set():
                var[key].setlb(None)
                var[key].setub(None)
        
    def set_prediction(self,results):
        measured_state = {}
        for x in self.states:
            for j in self.x_vars[x]:
                measured_state[(x,j)] = results[(x,(1,3)+j+(1,))]
        for y in self.y:
            for j in self.y_vars[y]:
                measured_state[(y,j)] = results[(y,(1,3)+j+(1,))]
        self.measurement[self.iterations] = deepcopy(measured_state)   #!!o!!

    
    def create_measurement(self,results,var_dict):
        """ var_dict = {(x.name,(add_indices)):relative variance for measurement noise}
            results = {(x.name,(add_indices)):results usually obtained by plant simulation}
        """
        # creates measurement for all variables that are in var_dict        
        # prediction for these variables must have been generated beforehand 
        # due to set_prediction only allowing for states --> ALL measured variables must be states
        
        # Sets possibly more measured states than are loaded into the state estimation problem
        # what is specified in self.y determines which states are considered as measured for estimation
        noise_init = {}
        measured_state = {}
        for key in var_dict:
                x = key[0]
                j = key[1] + (1,)
                noise_init[key] = np.random.normal(loc=0.0, scale=var_dict[key])
                measured_state[key] = (1-noise_init[key])*results[(x,(1,3)+j)]
        self.measurement[self.iterations] = deepcopy(measured_state)
        
        # update values in lsmhe problem
        for y in self.y:
            for j in self.y_vars[y]:
                vni = self.yk_key[(y,j)]
                self.lsmhe.yk0_mhe[self.nfe_mhe,vni] = self.measurement[self.nfe_mhe][(y,j)]
                
    def cycle_mhe(self,initialguess,m_cov,q_cov,u_cov,first_call=False):
        # open the parameters as degree of freedom
        if self.noisy_params: 
            self.lsmhe.par_to_var()
            for p in self.p_noisy:
                p_mhe = getattr(self.lsmhe,p)
                for key in self.p_noisy[p]:
                    p_mhe[key].unfix()
        
        # load initialguess from previous iteration lsmhe and olnmpc for the missing element
        if first_call:
            for _var in self.lsmhe.component_objects(Var):
                for _key in _var.index_set():
                    try:
                        _var[_key].value = initialguess[(_var.name,_key)]
                    except KeyError:
                        try:
                            if type(_key) == int:
                                aux_key = (_key,) + (1,)
                            elif _key == None:
                                continue
                            else:
                                aux_key = (1,) + _key[1:] + (1,)
                            _var[_key].value = initialguess[(_var.name,aux_key)] # xxx
                        except KeyError: # fallback strategy
                            continue
                        except AttributeError: # fallback strategy
                            continue
        else:
            for _var in self.lsmhe.component_objects(Var):
                for _key in _var.index_set():
                    try:
                        _var[_key].value = initialguess[(_var.name,_key)]
                    except KeyError:
                        try:
                            var_nmpc = getattr(self.olnmpc, _var.name)
                            if type(_key) == int:
                                aux_key = (_key,) + (1,)
                            elif _key == None:
                                continue
                            else:
                                aux_key = (1,) + _key[1:] + (1,)
                            _var[_key].value = var_nmpc[aux_key].value # xxx
                        except KeyError: # fallback strategy
                            continue
                        except AttributeError: # fallback strategy
                            continue
                
            
        # adjust the time intervals via model parameter self.lsmhe.fe_dist[i]
        self.lsmhe.tf.fixed = True
        self.lsmhe.tf.value = self.recipe_optimization_model.tf[1,1].value # base is set via recipe_optimization_model
        for i in self.lsmhe.fe_t:
            self.lsmhe.fe_dist[i] = (self.nmpc_trajectory[i,'tf']-self.nmpc_trajectory[i-1,'tf'])/self.lsmhe.tf.value
        
        # fix the applied controls in the model:    
        # self.lsmhe.equalize_u(direction='r_to_u')
        for u in self.u:
            control = getattr(self.lsmhe, u)
            control[self.nfe_mhe].value = self.curr_u[u]
            control.fix()
            #for i in control.index_set():
            #    control[i].value = self.nmpc_trajectory[i,u]
            #    control[i].fixed = True
        
        # reapply the measured variables
        for t in self.lsmhe.fe_t:
            #for x in self.x_noisy: # should be whatever is measured not necessarily noisy states
            for x in self.y:
                #for j in self.state_vars[x]:
                for j in self.y_vars[x]:
                    vni = self.yk_key[(x,j)]
                    self.lsmhe.yk0_mhe[t,vni] = self.measurement[t][(x,j)]
                    
        # redo the initialization of estimates and matrices
        self.set_covariance_meas(m_cov)
        self.set_covariance_disturb(q_cov)
        self.set_covariance_u(u_cov)
        
        # activate whats necessary + leggo:
        self.lsmhe.obfun_mhe.activate() # objective function!
        self.lsmhe.noisy_cont.activate() # noisy constraints!
        self.lsmhe.hyk_c_mhe.activate()
        self.lsmhe.eobj.deactivate()
        if self.noisy_inputs:
            for u in self.u:
                con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
                con_w.activate()
            
        
    def solve_mhe(self,fix_noise=False):
        if fix_noise:
            self.lsmhe.wk_mhe.fix()
        # solve statement
        for i in self.lsmhe.eps.index_set():
            self.lsmhe.eps[i].value = 0
        self.lsmhe.eps.fix()
        
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "no"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 3000
        with open("ipopt.opt", "w") as f:
            f.write("print_info_string yes")
            f.close()

        result = ip.solve(self.lsmhe, tee=True)
        aux = value(self.lsmhe.obfun_mhe)
        #for k in self.lsmhe.wk_mhe.index_set():
        #    self.lsmhe.wk_mhe[k] = 0
        #    self.lsmhe.wk_mhe[k].fixed = False 
        
        #result = ip.solve(self.lsmhe, tee=True)
        #aux =- value(self.lsmhe.obfun_mhe)
        
        output = self.store_results(self.lsmhe) # saves the results to initialize the upcoming mhe problem
    
        # saves the predicted initial_values for the states
        for x in self.states:
            for j in self.x_vars[x]:
                self.initial_values[(x,j+(1,))] = output[(x,(self.nfe_mhe,3)+j)]
                self.current_state_info[(x,j)] = self.initial_values[(x,j+(1,))]# !!o!! set current_state_info as the estimated state
                    # eliminate one ? save the same purpose
        # expand the nfe_mhe horizon
        #self.nfe_mhe += 1 this is done by cycle_iteration --> easier to switch from as to normal nmpc/mhe
        
        # load solution status in self.nmpc_trajectory
        self.nmpc_trajectory[self.iterations,'solstat_mhe'] = [str(result.solver.status),str(result.solver.termination_condition)]
        self.nmpc_trajectory[self.iterations,'obj_value_mhe'] = aux
        return output   
        
    def cycle_ics_mhe(self, nmpc_as = False, mhe_as = False):
        if nmpc_as and mhe_as: # both nmpc and mhe use advanced step schme
            ivs = self.initial_values
        elif nmpc_as: # only nmpc uses advanced step scheme
            ivs = self.curr_pstate
        else: # nothing uses advanced step scheme
            ivs = self.initial_values
            
        for x in self.states:
            xic = getattr(self.olnmpc, x+'_ic')
            for j in self.x_vars[x]:
                if not(j == ()):
                    xic[j].value = ivs[(x,j+(1,))]
                else:
                    xic.value = ivs[(x,j+(1,))]
                    
        if self.noisy_params and self.iterations > 1:
            ##################################################################
            # comute principle components of approximate 95%-confidence region
            ##################################################################
            try: # adapt parameters iff estimates are confident
                dimension = int(np.sqrt(len(self.mhe_confidence_ellipsoids[1])))
                confidence = chi2.isf(1-0.95,dimension)
                A_dict = self.mhe_confidence_ellipsoids[self.iterations-1]
                # assemble dimension x dimension array
                rows = {}
                for m in range(dimension):
                        rows[m] = np.array([A_dict[(m,i)] for i in range(dimension)])
                A = 1/confidence*np.array([np.array(rows[i]) for i in range(dimension)]) # shape matrix of confidence ellipsoid
                #A = np.linalg.inv(A)
                U, s, V = np.linalg.svd(A) # singular value decomposition of shape matrix 
                radii = 1/np.sqrt(s) # radii --
                
                if self.adapt_params:
                    for p in self.p_noisy:
                        p_nom = getattr(self.olnmpc,p)
                        p_mhe = getattr(self.lsmhe,p)
                        p_fs = getattr(self.forward_simulation_model,p)
                        for key in self.p_noisy[p]:
                            index = self.PI_indices[p,key]
                            dev = -1e8
                            for m in range(dimension):
                                 dev = max(dev,(abs(radii[m]*U[index][m]) + p_mhe[key].value)/p_mhe[key].value)
                            if dev < 1 + self.estimate_acceptance:
                                p_nom[key].value = p_mhe[key].value
                                p_fs[key].value = p_mhe[key].value
                            else:
                                continue
            except KeyError: # adapt parameters blindly
                if self.adapt_params:
                    for p in self.p_noisy:
                        p_nom = getattr(self.olnmpc,p)
                        p_mhe = getattr(self.lsmhe,p)
                        p_fs = getattr(self.forward_simulation_model,p)
                        for key in self.p_noisy[p]:
                             p_nom[key].value = p_mhe[key].value
                             p_fs[key].value = p_mhe[key].value
            
            # adapt parameters iff estimates are confident enough
            if self.adapt_params:
                for p in self.p_noisy:
                    p_nom = getattr(self.olnmpc,p)
                    p_mhe = getattr(self.lsmhe,p)
                    p_fs = getattr(self.forward_simulation_model,p)
                    for key in self.p_noisy[p]:
                        index = self.PI_indices[p,key]
                        dev = -1e8
                        for m in range(dimension):
                             dev = max(dev,(abs(radii[m]*U[index][m]) + p_mhe[key].value)/p_mhe[key].value)
                        if dev < 1 + self.confidence_threshold:
                            p_nom[key].value = p_mhe[key].value
                            p_fs[key].value = p_mhe[key].value
                        else:
                            continue
            ###############################################################
            ### DISCLAIMER:
            ### currently tailored to single stage which is reasonable since multiple stages do not make sense
            ###############################################################
            if self.update_scenario_tree:
                # idea: go through the axis of the ellipsoid (U) and include the intersections of this axis with confidence ellipsoid on both ends as scenarios (sigmapoints)
                # only accept these scenarios if sigmapoints are inside hypercube spanned by euclidean unit vectors around nominal value  
                l = 0
                flag=False
                for m in range(dimension):
                    l += 2
                    for p in self.p_noisy:
                        p_scen = getattr(self.olnmpc,'p_'+p)
                        p_nom = getattr(self.olnmpc,p)
                        p_mhe = getattr(self.lsmhe,p)
                        for key in self.p_noisy[p]:
                            index = self.PI_indices[p_mhe.name,key]
                            dev = -1e8
                            for m in range(dimension): # little redundant but ok
                                dev = max(dev,(abs(radii[m]*U[index][m]) + p_mhe[key].value)/p_mhe[key].value)
                            if dev < 1 + self.confidence_threshold:# confident enough in parameter estimate --> adapt parameter in prediction and NMPC model
                                if dev > 1 + self.robustness_threshold:# minimum robustness threshold is not reached
                                    for t in self.olnmpc.fe_t:
                                        p_scen[(key,t,l)].value = (radii[m]*U[index][m] + p_nom[key].value)/p_nom[key].value
                                        p_scen[(key,t,l+1)].value = (p_nom[key].value - radii[m]*U[index][m])/p_nom[key].value
                                else:# minimum robustness threshold is reached already
                                    if np.sign(U[index][m]) == 1:
                                        for t in self.olnmpc.fe_t:
                                            p_scen[(key,t,l)].value = 1+self.robustness_threshold
                                            p_scen[(key,t,l+1)].value = 1-self.robustness_threshold
                                    else:
                                        for t in self.olnmpc.fe_t:
                                            p_scen[(key,t,l)].value = 1-self.robustness_threshold
                                            p_scen[(key,t,l+1)].value = 1+self.robustness_threshold
                            else:
                                flag = True
                                break
                        if flag:
                            break
                    if flag:
                        break
                    
                # if flag=True sigmapoints not inside basic hypercube --> use corners of hypercube instead
                if flag:
                    # set all values ot 1 
                    for p in self.p_noisy:
                        p_scen = getattr(self.olnmpc, 'p_'+p)
                        for key in p_scen.index_set():
                            p_scen[key] = 1.0
                            
                    # keep using base case scenarios
                    l = 2
                    for p in self.p_noisy:
                        p_scen = getattr(self.olnmpc,'p_'+p)
                        for key in self.p_noisy[p]:
                            for t in self.olnmpc.fe_t:
                                p_scen[(key,t,l)].value = 1 + self.confidence_threshold
                                p_scen[(key,t,l+1)].value = 1 - self.confidence_threshold
                            l += 2
                                
                                    
                                    
                    # ALTERNATIVE: ONLY USE THE LONGEST HALF AXIS
                    #for p in self.p_noisy:
                    #    p_scen = getattr(self.olnmpc,'p_'+p)
                    #    p_nom = getattr(self.olnmpc,p)
                    #    p_mhe = getattr(self.lsmhe,p)
                    #    for key in self.p_noisy[p]:
                    #        index = self.PI_indices[p_mhe.name,key]
                    #        ex1 = (radii[0]*U[index][0] + p_nom[key].value)/p_nom[key].value 
                    #        ex2 = (p_nom[key].value - radii[0]*U[index][0])/p_nom[key].value
                    #        for i in self.olnmpc.fe_t:
                    #            p_scen[(key,i,2)].value = ex1 
                    #            p_scen[(key,i,3)].value = ex2

    def compute_offset_measurements(self):
        mhe_y = getattr(self.lsmhe, "yk0_mhe")
        for y in self.y:
            for j in self.y_vars[y]:
                k = self.yk_key[(y, j)]
                mhe_yval = value(mhe_y[self.nfe_mhe, k])
                self.curr_y_offset[(y, j)] = mhe_yval - self.measurement[self.iterations][(y,j)]
                
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    def initialize_xreal(self, ref):
        """Wanted to keep the states in a horizon-like window, this should be done in the main dyngen class"""
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy [xreal]"
        self.load_d_d(ref, dum, 1)
        for fe in range(1, self._window_keep):
            for i in self.states:
                pn = i + "_ic"
                p = getattr(dum, pn)
                vs = getattr(dum, i)
                for ks in p.iterkeys():
                    p[ks].value = value(vs[(1, self.ncp_t) + (ks,)])
            #: Solve
            self.solve_d(dum, o_tee=False)
            for i in self.states:
                self.xreal_W[(i, fe)] = []
                xs = getattr(dum, i)
                for k in xs.keys():
                    if k[1] == self.ncp_t:
                        print(i)
                        self.xreal_W[(i, fe)].append(value(xs[k]))

    def init_lsmhe_prep(self, ref):
        """Initializes the lsmhe in preparation phase
        Args:
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model"""
        self.journalizer("I", self._c_it, "initialize_lsmhe", "Attempting to initialize lsmhe")
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy I"
        #: Load current solution
        self.load_d_d(ref, dum, 1)
        #: Patching of finite elements
        for finite_elem in range(1, self.nfe_t + 1):
            #: Cycle ICS
            for i in self.states:
                pn = i + "_ic"
                p = getattr(dum, pn)
                vs = getattr(dum, i)
                for ks in p.iterkeys():
                    p[ks].value = value(vs[(1, self.ncp_t) + (ks,)])
            if finite_elem == 1:
                for i in self.states:
                    pn = i + "_ic"
                    p = getattr(self.lsmhe, pn)  #: Target
                    vs = getattr(dum, i)  #: Source
                    for ks in p.iterkeys():
                        p[ks].value = value(vs[(1, self.ncp_t) + (ks,)])
            self.patch_meas_mhe(finite_elem, src=self.d1)
            #: Solve
            self.solve_d(dum, o_tee=False)
            #: Patch
            self.load_d_d(dum, self.lsmhe, finite_elem)
            self.load_input_mhe("mod", src=dum, fe=finite_elem)
        tst = self.solve_d(self.lsmhe, o_tee=False, skip_update=False)
        if tst != 0:
            sys.exit()
        for i in self.x_noisy:
            cp_con = getattr(self.lsmhe, "cp_" + i)
            cp_con.deactivate()
        self.lsmhe.noisy_cont.activate()

        self.lsmhe.obfun_dum_mhe_deb.deactivate()
        self.lsmhe.obfun_dum_mhe.activate()

        self.lsmhe.hyk_c_mhe.activate()
        for u in self.u:
            cc = getattr(self.lsmhe, u + "_c")  #: Get the constraint for input
            con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
            cc.deactivate()
            con_w.activate()

        # if self.deact_ics:
        #     for i in self.states:
        #         self.lsmhe.del_component(i + "_icc")

        self.journalizer("I", self._c_it, "initialize_lsmhe", "Attempting to initialize lsmhe Done")

    def patch_meas_mhe(self, t, **kwargs):
        """Mechanism to assign a value of y0 to the current mhe from the dynamic model
        Args:
            t (int): int The current collocation point
        Returns:
            meas_dict (dict): A dictionary containing the measurements list by meas_var
        """
        src = kwargs.pop("src", None)
        skip_update = kwargs.pop("skip_update", False)
        noisy = kwargs.pop("noisy", True)

        meas_dic = dict.fromkeys(self.y)
        l = []
        for i in self.y:
            lm = []
            var = getattr(src, i)
            for j in self.y_vars[i]:
                lm.append(value(var[(1, self.ncp_t,) + j]))
                l.append(value(var[(1, self.ncp_t,) + j]))
            meas_dic[i] = lm

        if not skip_update:  #: Update the mhe model
            self.journalizer("I", self._c_it, "patch_meas_mhe", "Measurement patched to " + str(t))
            y0dest = getattr(self.lsmhe, "yk0_mhe")
            # print("there is an update", file=sys.stderr)
            for i in self.y:
                for j in self.y_vars[i]:
                    k = self.yk_key[(i, j)]
                    #: Adding noise to the mhe measurement
                    y0dest[t, k].value = l[k] + self.curr_m_noise[(i, j)] if noisy else l[k]
        return meas_dic

    def adjust_nu0_mhe(self):
        """Adjust the initial guess for the nu variable"""
        for t in self.lsmhe.fe_t:
            k = 0
            for i in self.y:
                for j in self.y_vars[i]:
                    target = value(self.lsmhe.yk0_mhe[t, k]) - value(self.yk_l[t][k])
                    self.lsmhe.nuk_mhe[t, k].set_value(target)
                    k += 1

    def adjust_w_mhe(self):
        for i in range(1, self.nfe_t):
            j = 0
            for x in self.x_noisy:
                x_var = getattr(self.lsmhe, x)
                for k in self.x_vars[x]:
                    x1pvar_val = value(x_var[(i+1, 0), k])
                    x1var_val = value(x_var[(i, self.ncp_t), k])
                    self.lsmhe.wk_mhe[i, j].set_value(x1pvar_val - x1var_val)
                    j += 1

    def set_covariance_meas(self, cov_dict):
        """Sets covariance(inverse) for the measurements.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(meas_name, j), (meas_name, k), time]
        Returns:
            None
        """
        # measured states R_mhe
        rtarget = getattr(self.lsmhe, "R_mhe")
        for key in cov_dict:
            # 1. check whether the variable is even a measured one
            if key[0] in self.yk_key: 
                # 2. if yes: compute the respective weight as follows:
                vni = key[0]
                vnj = key[1]
    #            _t = key[2]
                v_i = self.yk_key[vni]
                v_j = self.yk_key[vnj]
                # try:
                for _t in range(1,self.nfe_mhe+1):
                    if self.diag_Q_R:                
                        try:
                            rtarget[_t, v_i] = 1 / (cov_dict[vni, vnj]*self.measurement[_t][vni] + 0.001)**2
                        except ZeroDivisionError:
                            rtarget[_t,v_i] = 1 
                    else:
                        rtarget[_t, v_i, v_j] = cov_dict[vni, vnj]
            else:
                continue
                # except KeyError:
            #     print("Key error, {:} {:} {:}".format(vni, vnj, _t))

    def set_covariance_disturb(self, cov_dict, set_bounds=True):
        """Sets covariance(inverse) for the states.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(state_name, j), (state_name, k), time]
        Returns:
            None
        """
        # disturbances
        qtarget = getattr(self.lsmhe, "Q_mhe")
        w = getattr(self.lsmhe, 'wk_mhe')
        for key in cov_dict:
            if key[0] in self.xkN_key:
                vni = key[0]
                vnj = key[1]
                v_i = self.xkN_key[vni]
                v_j = self.xkN_key[vnj]
            
                xic = getattr(self.lsmhe, vni[0] + "_ic")
                try:
                    if vni[1] == ():
                        qtarget[0, v_i] = 1 / (cov_dict[vni, vnj]*xic.value + 0.001)**2 # .00001
                    else:
                        qtarget[0, v_i] = 1 / (cov_dict[vni, vnj]*xic[vni[1]].value + 0.001)**2 # .0001
                    
                    if set_bounds:
                        if cov_dict[vni, vnj] != 0:
                            if vni[1] == (): 
                                confidence = 3*cov_dict[vni, vnj]*xic.value
                            else:
                                confidence = 3*cov_dict[vni, vnj]*xic[vni[1]].value
                        else:
                            if vni[1] == ():     
                                confidence = 0.5*xic.value
                            else:
                                confidence = 0.5*xic[vni[1]].value
                        w[0,v_i].setlb(-confidence)
                        w[0,v_i].setub(confidence)
                        
                        if w[0,v_i].lb == w[0,v_i].ub:
                            w[0,v_i].fix()

                except ZeroDivisionError:
                    qtarget[0, v_i] = 1 
                    # t > 1 is systematic:
                for _t in range(1,self.nfe_mhe):
                    if self.diag_Q_R:
                        try:
                            qtarget[_t, v_i] = 1 / (cov_dict[vni, vnj]*self.nmpc_trajectory[_t,vni] + .001)**2 # 0.00001
                        except ZeroDivisionError:
                            qtarget[_t, v_i] = 1
                    else:
                        qtarget[_t, v_i, v_j] = cov_dict[vni, vnj]
                    
                    # bound disturbances to help solution
                    if set_bounds:                        
                        if cov_dict[vni, vnj] != 0.0:
                            confidence = 3*cov_dict[vni,vnj]*self.nmpc_trajectory[_t,vni]
                        else:
                            confidence = 0.5*self.nmpc_trajectory[_t,vni]

                        w[_t,v_i].setlb(-confidence)
                        w[_t,v_i].setub(confidence)
                        
                        if abs(w[_t,v_i].lb - w[_t,v_i].ub) < 1e-5:
                            w[_t,v_i].fix()
            

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
                if _t > self.nfe_t_0:
                    aux_key = self.nfe_t_0 # 
                else:
                    aux_key = (_t,1) # tailored to my code basically
                try:
                    utarget[_t, vni] = 1 / (cov_dict[key]*self.reference_control_trajectory[vni,aux_key] + .001)**2
                except ZeroDivisionError:
                    utarget[_t, vni] = 1
                #qtarget[_t, vni] = 1 / cov_dict[key]

    def shift_mhe(self):
        """Shifts current initial guesses of variables for the mhe problem"""
        for v in self.lsmhe.component_objects(Var, active=True):
            if type(v.index_set()) == SimpleSet:  #: Don't want simple sets
                continue
            else:
                kl = v.keys()
                if len(kl[0]) < 2:
                    continue
                for k in kl:
                    if k[0] < self.nfe_t:
                        try:
                            v[k].set_value(v[(k[0] + 1,) + k[1:]])
                        except ValueError:
                            continue

    def shift_measurement_input_mhe(self):
        """Shifts current measurements for the mhe problem"""
        y0 = getattr(self.lsmhe, "yk0_mhe")
        for i in range(2, self.nfe_t + 1):
            for j in self.lsmhe.yk0_mhe.keys():
                y0[i-1, j[1:]].value = value(y0[i, j[1:]])
            for u in self.u:
                umhe = getattr(self.lsmhe, u)
                umhe[i-1] = value(umhe[i])
        self.adjust_nu0_mhe()


    def load_input_mhe(self, src_kind, **kwargs):
        """Loads inputs into the mhe model"""
        src = kwargs.pop("src", self.d1)
        fe = kwargs.pop("fe", 1)
        # src_kind = kwargs.pop("src_kind", "mod")
        if src_kind == "mod":
            for u in self.u:
                usrc = getattr(src, u)
                utrg = getattr(self.lsmhe, u)
                utrg[fe].value = value(usrc[1])
        elif src_kind == "self.dict":
            for u in self.u:
                utrg = getattr(self.lsmhe, u)
                utrg[fe].value = value(self.curr_u[u])

    def init_step_mhe(self, tgt, i, patch_pred_y=False):
        """Takes the last state-estimate from the mhe to perform an open-loop simulation
        that initializes the last slice of the mhe horizon
        Args:
            tgt (pyomo.core.base.PyomoModel.ConcreteModel): The target model
            i (int): finite element of lsmhe
            patch_y (bool): If true, patch the measurements as well"""
        src = self.lsmhe
        for vs in src.component_objects(Var, active=True):
            if vs.getname()[-4:] == "_mhe":
                continue
            vd = getattr(tgt, vs.getname())
            # there are two cases: 1 key 1 elem, several keys 1 element
            vskeys = vs.keys()
            if len(vskeys) == 1:
                #: One key
                for ks in vskeys:
                    for v in vd.itervalues():
                        v.set_value(value(vs[ks]))
            else:
                k = 0
                for ks in vskeys:
                    if k == 0:
                        if type(ks) != tuple:
                            #: Several keys of 1 element each!!
                            vd[1].set_value(value(vs[vskeys[-1]]))  #: This has got to be true
                            break
                        k += 1
                    kj = ks[2:]
                    if vs.getname() in self.states:  #: States start at 0
                        for j in range(0, self.ncp_t + 1):
                            vd[(1, j) + kj].set_value(value(vs[(i, j) + kj]))
                    else:
                        for j in range(1, self.ncp_t + 1):
                            vd[(1, j) + kj].set_value(value(vs[(i, j) + kj]))
        for u in self.u:  #: This should update the inputs
            usrc = getattr(src, u)
            utgt = getattr(tgt, u)
            utgt[1] = (value(usrc[i]))
        for x in self.states:
            pn = x + "_ic"
            p = getattr(tgt, pn)
            vs = getattr(self.lsmhe, x)
            for ks in p.iterkeys():
                p[ks].value = value(vs[(i, self.ncp_t) + (ks,)])
        self.solve_d(tgt, o_tee=False, stop_if_nopt=True)
        self.load_d_d(tgt, self.lsmhe, self.nfe_t)

        if patch_pred_y:
            self.journalizer("I", self._c_it, "init_step_mhe", "Prediction for advanced-step.. Ready")
            self.patch_meas_mhe(self.nfe_t, src=tgt, noisy=True)
        self.adjust_nu0_mhe()
        self.adjust_w_mhe()
    
    def compute_confidence_ellipsoid(self):
        """ computes confidence ellipsoids for estimated parameters via 1st order approximation"""
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
        
        # FIX THAT ASAP
        try:
            try:
                self.k_aug.solve(self.lsmhe, tee=True)
            except ApplicationError:
                self.nmpc_trajectory[self.iterations,'solstat_mhe'] = ['Inversion of Reduced Hessian failed','Inversion of Reduced Hessian failed']
        except NameError:
            self.nmpc_trajectory[self.iterations,'solstat_mhe'] = ['Inversion of Reduced Hessian failed','Inversion of Reduced Hessian failed']
            
            
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
        
        self.mhe_confidence_ellipsoids[self.iterations] = deepcopy(self._PI)


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
                for j in self.x_vars[key]:
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
                for j in self.x_vars[key]:
#                    var[(self.nfe_t, self.ncp_t) + j].set_suffix_value(self.lsmhe.dof_v, 1)
                    var[(self.nfe_mhe, self.ncp_t) + j].set_suffix_value(self.lsmhe.dof_v, 1)

    def check_active_bound_noisy(self):
        """Checks if the dof_(super-basic) have active bounds, if so, add them to the exclusion list"""
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name

        self.xkN_nexcl = []
        k = 0
        for x in self.x_noisy:
            v = getattr(self.lsmhe, x)
            for j in self.x_vars[x]:
                active_bound = False
                if v[(2, 0) + j].lb:
                    if v[(2, 0) + j].value - v[(2, 0) + j].lb < 1e-08:
                        active_bound = True
                if v[(2, 0) + j].ub:
                    if v[(2, 0) + j].ub - v[(2, 0) + j].value < 1e-08:
                        active_bound = True
                if active_bound:
                    print("Active bound {:s}, {:d}, value {:f}".format(x, j[0], v[(2, 0) + j].value), file=sys.stderr)
                    v[(2, 0) + j].set_suffix_value(self.lsmhe.dof_v, 0)
                    self.xkN_nexcl.append(0)
                    k += 1
                else:
                    v[(2, 0) + j].set_suffix_value(self.lsmhe.dof_v, 1)
                    self.xkN_nexcl.append(1)  #: Not active, add it to the non-exclusion list.
        if k > 0:
            print("I[[check_active_bound_noisy]] {:d} Active bounds.".format(k))

    def deact_icc_mhe(self):
        """Deactivates the icc constraints in the mhe problem"""
        if self.deact_ics:
            for i in self.states:
                try:
                    icccon = getattr(self.lsmhe, i + "_icc")
                    self.lsmhe.del_component(icccon)
                except AttributeError:
                    continue


        #: Maybe only for a subset of the states
        else:
            for i in self.states:
                if i in self.x_noisy:
                    ic_con = getattr(self.lsmhe, i + "_icc")
                    for k in ic_con.keys():
                        if k[2:] in self.x_vars[i]:
                            ic_con[k].deactivate()

    def regen_objective_fun(self):
        """Given the exclusion list, regenerate the expression for the arrival cost"""
        self.lsmhe.Arrival_e_mhe.set_value(0.5 * sum((self.xkN_l[j] - self.lsmhe.x_0_mhe[j]) *
                                                     sum(self.lsmhe.PikN_mhe[j, k] *
                                                         (self.xkN_l[k] - self.lsmhe.x_0_mhe[k]) for k in
                                                         self.lsmhe.xkNk_mhe if self.xkN_nexcl[k])
                                                     for j in self.lsmhe.xkNk_mhe if self.xkN_nexcl[j]))
        self.lsmhe.obfun_mhe.set_value(self.lsmhe.Arrival_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.R_e_mhe)
        if self.lsmhe.obfun_dum_mhe.active:
            self.lsmhe.obfun_dum_mhe.deactivate()
        if not self.lsmhe.obfun_mhe.active:
            self.lsmhe.obfun_mhe.activate()

    def load_covariance_prior(self):
        """Computes the reduced-hessian (inverse of the prior-covariance)
        Reads the result_hessian.txt file that contains the covariance information"""
        self.journalizer("I", self._c_it, "load_covariance_prior", "K_AUG w red_hess")
        self.k_aug.options["compute_inv"] = ""
        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        self.create_rh_sfx()
        self.k_aug.solve(self.lsmhe, tee=True)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        self._PI.clear()
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

    def set_state_covariance(self):
        """Sets covariance(inverse) for the prior_state.
        Args:
            None
        Return:
            None
        """
        pikn = getattr(self.lsmhe, "PikN_mhe")
        for key_j in self.x_noisy:
            for key_k in self.x_noisy:
                vj = getattr(self.lsmhe, key_j)
                vk = getattr(self.lsmhe, key_k)
                for j in self.x_vars[key_j]:
                    if vj[(2, 0) + j].get_suffix_value(self.lsmhe.dof_v) == 0:
                        #: This state is at its bound, skip
                        continue
                    for k in self.x_vars[key_k]:
                        if vk[(2, 0) + k].get_suffix_value(self.lsmhe.dof_v) == 0:
                            #: This state is at its bound, skip
                            print("vj {:s} {:d} .sfx={:d}, vk {:s} {:d}.sfx={:d}"
                                  .format(key_j, j[0], vj[(2, 0) + j].get_suffix_value(self.lsmhe.dof_v),
                                          key_k, k[0], vk[(2, 0) + k].get_suffix_value(self.lsmhe.dof_v),))
                            continue
                        row = vj[(2, 0) + j].get_suffix_value(self.lsmhe.rh_name)
                        col = vk[(2, 0) + k].get_suffix_value(self.lsmhe.rh_name)
                        #: Ampl does not give you back 0's
                        if not row:
                            row = 0
                        if not col:
                            col = 0

                        # print((row, col), (key_j, j), (key_k, k))
                        q0j = self.xkN_key[key_j, j]
                        q0k = self.xkN_key[key_k, k]
                        pi = self._PI[row, col]
                        try:
                            pikn[q0j, q0k] = pi
                        except KeyError:
                            errk = key_j + "_" + str(j) + ", " + key_k + "_" + str(k)
                            print("Kerror, var {:}".format(errk))
                            pikn[q0j, q0k] = 0.0

    def set_prior_state_from_prior_mhe(self):
        """Mechanism to assign a value to x0 (prior-state) from the previous mhe
        Args:
            None
        Returns:
            None
        """
        for x in self.x_noisy:
            var = getattr(self.lsmhe, x)
            for j in self.x_vars[x]:
                z0dest = getattr(self.lsmhe, "x_0_mhe")
                z0 = self.xkN_key[x, j]
                z0dest[z0] = value(var[(2, 0,) + j])

    def update_noise_meas(self, mod, cov_dict):
        self.journalizer("I", self._c_it, "introduce_noise_meas", "Noise introduction")
        # f = open("m0.txt", "w")
        # f1 = open("m1.txt", "w")
        for y in self.y:
            vy = getattr(mod,  y)
            # vy.display(ostream=f)
            for j in self.y_vars[y]:
                vv = value(vy[(1, self.ncp_t) + j])
                sigma = cov_dict[(y, j), (y, j), 1]
                self.curr_m_noise[(y, j)] = np.random.normal(0, sigma)
                # noise = np.random.normal(0, sigma)
                # # print(noise)
                # vv += noise
                # vy[(1, self.ncp_t) + j].set_value(vv)
            # vy.display(ostream=f1)
        # f.close()
        # f1.close()

    def print_r_mhe(self):
        self.journalizer("I", self._c_it, "print_r_mhe", "Results at" + os.getcwd())
        self.journalizer("I", self._c_it, "print_r_mhe", "Results suffix " + self.res_file_suf)
        for x in self.x_noisy:
            elist = []
            rlist = []
            xe = getattr(self.lsmhe, x)
            xr = getattr(self.d1, x)
            for j in self.x_vars[x]:
                elist.append(value(xe[(self.nfe_t, self.ncp_t) + j]))
                rlist.append(value(xr[(1, self.ncp_t) + j]))
            self.s_estimate[x].append(elist)
            self.s_real[x].append(rlist)

        # with open("res_mhe_ee.txt", "w") as f:
        #     for x in self.x_noisy:
        #         for j in range(0, len(self.s_estimate[x][0])):
        #             for i in range(0, len(self.s_estimate[x])):
        #                 xvs = str(self.s_estimate[x][i][j])
        #                 f.write(xvs)
        #                 f.write('\t')
        #             f.write('\n')
        #     f.close()

        with open("res_mhe_es_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.x_noisy:
                for j in self.s_estimate[x][-1]:
                    xvs = str(j)
                    f.write(xvs)
                    f.write('\t')
            f.write('\n')
            f.close()
        with open("res_mhe_rs_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.x_noisy:
                for j in self.s_real[x][-1]:
                    xvs = str(j)
                    f.write(xvs)
                    f.write('\t')
            f.write('\n')
            f.close()
        with open("res_mhe_eoff_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.x_noisy:
                for j in range(0, len(self.s_estimate[x][-1])):
                    e = self.s_estimate[x][-1][j]
                    r = self.s_real[x][-1][j]
                    xvs = str(e-r)
                    f.write(xvs)
                    f.write('\t')
            f.write('\n')
            f.close()
        # with open("res_mhe_ereal.txt", "w") as f:
        #     for x in self.x_noisy:
        #         for j in range(0, len(self.s_real[x][0])):
        #             for i in range(0, len(self.s_real[x])):
        #                 xvs = str(self.s_real[x][i][j])
        #                 f.write(xvs)
        #                 f.write('\t')
        #             f.write('\n')
        #     f.close()

        for y in self.y:
            elist = []
            rlist = []
            nlist = []
            yklst = []
            ye = getattr(self.lsmhe, y)
            yr = getattr(self.d1, y)
            for j in self.y_vars[y]:
                elist.append(value(ye[(self.nfe_t, self.ncp_t) + j]))
                rlist.append(value(yr[(1, self.ncp_t) + j]))
                nlist.append(self.curr_m_noise[(y, j)])
                yklst.append(value(self.lsmhe.yk0_mhe[self.nfe_t, self.yk_key[(y, j)]]))
            self.y_estimate[y].append(elist)
            self.y_real[y].append(rlist)
            self.y_noise_jrnl[y].append(nlist)
            self.yk0_jrnl[y].append(yklst)

        # with open("res_mhe_ey.txt", "w") as f:
        #     for y in self.y:
        #         for j in range(0, len(self.y_estimate[y][0])):
        #             for i in range(0, len(self.y_estimate[y])):
        #                 yvs = str(self.y_estimate[y][i][j])
        #                 f.write(yvs)
        #                 f.write('\t')
        #             f.write('\n')
        #     f.close()

        with open("res_mhe_ey_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.y_estimate[y][-1]:
                    yvs = str(j)
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_yreal_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.y_real[y][-1]:
                    yvs = str(j)
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_yk0_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.yk0_jrnl[y][-1]:
                    yvs = str(j)
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_ynoise_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.y_noise_jrnl[y][-1]:
                    yvs = str(j)
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_yoffset_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.y_vars[y]:
                    yvs = str(self.curr_y_offset[(y, j)])
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_unoise_" + self.res_file_suf + ".txt", "a") as f:
            for u in self.u:
                # u_mhe = getattr(self.lsmhe, u)
                ue_mhe = getattr(self.lsmhe, "w_" + u + "_mhe")
                for i in self.lsmhe.fe_t:
                    dv = value(ue_mhe[i])
                    dstr = str(dv)
                    f.write(dstr)
                    f.write('\t')
            f.write('\n')
            f.close()

    def compute_y_offset(self, noisy=True):
        mhe_y = getattr(self.lsmhe, "yk0_mhe")
        for y in self.y:
            plant_y = getattr(self.d1, y)
            for j in self.y_vars[y]:
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
            for j in self.y_vars[y]:
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
            for x in self.states:
                xvar = getattr(self.lsmhe, x)
                xic = getattr(self.olnmpc, x + "_ic")
                for j in self.state_vars[x]:
                    if j == ():
                        self.curr_state_offset[(x, j)] = value(xic)- value(xvar[self.nfe_mhe, self.ncp_t, j])
                    else:    
                        self.curr_state_offset[(x, j)] = value(xic[j])- value(xvar[self.nfe_mhe, self.ncp_t, j])
                    print("state !", self.curr_state_offset[(x, j)])

        for x in self.states:
            xvar = getattr(self.lsmhe, x)
            for j in self.x_vars[x]:
                self.curr_estate[(x, j + (1,))] = value(xvar[self.nfe_mhe, self.ncp_t, j])
                


    def method_for_mhe_simulation_step(self):
        pass

    def deb_alg_sys(self):
        """Debugging the algebraic system"""
        # Fix differential states
        # Deactivate ODEs de_
        # Deactivate FE cont cp_
        # Deactivate IC _icc
        # Deactivate coll dvar_t_

        # Deactivate hyk
        for i in self.x_noisy:
            x = getattr(self.lsmhe, i)
            x.fix()
            cp_con = getattr(self.lsmhe, "cp_" + i)
            cp_con.deactivate()
            de_con = getattr(self.lsmhe, "de_" + i)
            de_con.deactivate()
            icc_con = getattr(self.lsmhe, i + "_icc")
            icc_con.deactivate()
            dvar_con = getattr(self.lsmhe, "dvar_t_" + i)
            dvar_con.deactivate()

        self.lsmhe.obfun_dum_mhe.deactivate()
        self.lsmhe.obfun_dum_mhe_deb.activate()
        self.lsmhe.hyk_c_mhe.deactivate()
        self.lsmhe.noisy_cont.deactivate()

        for u in self.u:
            cc = getattr(self.lsmhe, u + "_c")  #: Get the constraint for input
            con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
            cc.deactivate()
            con_w.deactivate()

        # self.lsmhe.pprint(filename="algeb_mod.txt")
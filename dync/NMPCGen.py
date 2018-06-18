#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:51:51 2017

@author: flemmingholtorf
"""
#### 
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix, ConstraintList
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from main.dync.DynGen import DynGen
from six import iterkeys
from copy import deepcopy
import numpy as np
import sys, time, csv, resource

__author__ = "@FHoltorf"


class NMPCGen(DynGen):
    def __init__(self, **kwargs):
        DynGen.__init__(self, **kwargs)
        self.int_file_nmpc_suf = int(time.time())+1
        self.olnmpc = object() 

        # path constraints that shall be monitored
        self.path_constraints = kwargs.pop('path_constraints',[])
        
        # any additional information contained in model that shall be monitored
        self.model_info = kwargs.pop('model_info',{})
        
        # objective function type for optimal control problem
        self.obj_type = kwargs.pop('obj_type','tracking')
        
        # linapprox to robust optimal control
        self.linapprox = kwargs.pop('linapprox',False)
        
        # spec
        self.min_horizon = kwargs.pop('min_horizon',0)
        self.n_s = kwargs.pop('n_s',1) # number of scenarios

        self.noisy_model = self.d_mod(1,self.ncp_t)
        self.recipe_optimization_model = object() 
        # model to compute the reference trajectory (open loop optimal control problem with economic obj)
        
        self.reference_state_trajectory = {} # {('state',(fe,j)):value)}
        self.reference_control_profile = {} # {('control',(fe,j)):value)}
        self.storage_model = object()
        self.initial_values = {}
        self.nominal_parameter_values = {}
        self.delta_u = kwargs.pop('control_displacement_penalty',False)
        self.tf_bounds = kwargs.pop('tf_bounds',[10.0,30.0])
            
        # plant simulation model in order to distinguish between noise and disturbances
        self.plant_simulation_model = self.d_mod(1, self.ncp_t)
        self.plant_simulation_model.name = 'plant simulation model'
        self.plant_simulation_model.create_output_relations()
        self.plant_simulation_model.create_bounds()
        self.plant_simulation_model.del_pc_bounds()
        self.plant_simulation_model.deactivate_pc()
        self.plant_simulation_model.deactivate_epc()
        self.plant_simulation_model.eobj.deactivate()
        self.plant_trajectory = {} 
        self.plant_trajectory[0,'tf'] = 0
        # organized as follows   
        #- for states:   {i, ('name',(index)): value} # actual state at time step i
        #- for controls: {i, 'name': value} 
                # --> control input valid between iteration i-1 and i 
                #     (i=1 --> is the first piecewise constant control input) 
        
        self.nmpc_trajectory = {} 
        self.nmpc_trajectory[0,'tf'] = 0
        # organized as follows   
        #- for states:   {i, ('name',(index)): value} # state prediction at time step i
        #- for controls: {i, 'name': value}
                # --> control input valid between iteration i-1 and i 
                #     (i=1 --> is the first piecewise constant control input)  
        
        self.pc_trajectory = {} 
        # organized as follows 
        #  {(pc,(iteration,collocation_point):value}
        
        
        # for advanced step: 
        self.forward_simulation_model = self.d_mod(1, self.ncp_t) # only need 1 element for forward simulation
        self.forward_simulation_model.name = 'forward simulation model'
        self.forward_simulation_model.create_output_relations()
        self.forward_simulation_model.create_bounds()
        self.forward_simulation_model.deactivate_pc()
        self.forward_simulation_model.deactivate_epc()
        self.forward_simulation_model.eobj.deactivate()
        self.forward_simulation_model.del_pc_bounds()
        self.simulation_trajectory = {}
        # organized as follows   
        #- for states:   {i, ('name',(index)): value} # state prediction at time step i
        #- for controls: {i, 'name': value}
                # --> control input valid between iteration i-1 and i 
                #     (i=1 --> is the first piecewise constant control input)
      

        # parameter covariance matrix
        self.par_dict = kwargs.pop('par_dict',{}) #par_dict {'name of parameter':[list of indices]}
        
        # robust cl nmpc approximation
        # alpha represents the uncertainty level
        #   case a): same uncertainty level for all parameters (normalized on nominal value):
                        # --> alpha is a scalar: float
        #   case b): different uncertainty levels:
                        # --> alpha is dictionary/vector: {(parameter.name,(key)):value} or {(parameter.name,key):value}
        #   case c): uncertainty set is adapted based on the confidence ellipsoids
                        # --> put initial guess of confidence + keyword 'adapted' in tuple:  ({(parameter.nem,(key)):value},'adapted') 
                        # --> alpha is matrix that gets computed in MHEGen (represented as dictionary)
                        
        self.alpha = kwargs.pop('alpha',self.confidence_threshold)
     
        # 
        self.poi = kwargs.pop('poi',[]) # takes a list of variable names that shall be monitored
        self.monitor = {}
        
    """ setting reference trajectories for tracking-type objectives """
    def set_reference_state_trajectory(self,input_trajectory):
        # input_trajectory = {('state',(fe,cp,j)):value}
        self.reference_state_trajectory = input_trajectory

    def get_state_trajectory(self,d_mod):
        output = {}
        for x in self.x:
            x_var = getattr(d_mod,x)
            for key in x_var.index_set():
                if key[1] == self.ncp_t:
                    try:
                        output[(x,key)] = x_var[key].value
                    except KeyError:
                        print('something went wrong during get_state_trajectory')
                        continue
                else:
                    continue
        return output    
        
    def set_reference_control_profile(self,control_profile):
        # input_trajectory = {('state',(fe,j)):value} ---> this format
        self.reference_control_profile = control_profile
        
    def get_control_profile(self,d_mod):
        output = {}
        for control_name in self.u:
            _u = getattr(d_mod,control_name)
            for _key in _u.index_set():
                try:
                    output[(control_name,_key)] = _u[_key].value
                except KeyError:
                    print('something went wrong during get_state_trajectory')
                    continue
        return output        
    
    def set_regularization_weights(self, Q_w = 1.0, R_w = 1.0, K_w = 1.0):
        if self.obj_type == 'economic':
            for i in self.olnmpc.fe_t:
                self.olnmpc.K_w_nmpc[i] = K_w
        else:
            for i in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[i] = Q_w
                self.olnmpc.R_w_nmpc[i] = R_w
                self.olnmpc.K_w_nmpc[i] = K_w
                
        
    def load_reference_trajectories(self):
        # assign values of the reference trajectory to parameters 
        # self.olnmpc.xmpc_ref_nmpc and self.umpc_ref_nmpc
        
        # reference trajectory used to assign delta
        # tracking terms shifted by every iteration (therefore self._c_it+1)
        # first time this should be called is for i = 1 
        # --> only tracking terms for iterations >= 2 relevant since first 
        #     control input is determined based on recipe-optimization
        for x in self.x:
            for j in self.x[x]:
                for fe in range(1, self.nfe_t+1):
                    try:
                        self.olnmpc.xmpc_ref_nmpc[fe,self.xmpc_key[(x,j)]] = self.reference_state_trajectory[x,(fe+self._c_it,self.ncp_t)+j]
                        self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0/(max(abs(self.reference_state_trajectory[x,(fe+self._c_it,self.ncp_t)+j]), 1e-3))**2
                    except KeyError:
                        self.olnmpc.xmpc_ref_nmpc[fe,self.xmpc_key[(x,j)]] = self.reference_state_trajectory[x,(self.nfe_t_0,self.ncp_t)+j]
                        self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0/(max(abs(self.reference_state_trajectory[x,(self.nfe_t_0,self.ncp_t)+j]), 1e-3))**2

        
        for u in self.u:
            for fe in range(1, self.nfe_t+1):
                try:
                    self.olnmpc.umpc_ref_nmpc[fe,self.umpc_key[u]] = self.reference_control_profile[u,fe+self._c_it]
                    self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(max(abs(self.reference_control_profile[u,fe+self._c_it]),1e-3))**2
                except KeyError:
                    self.olnmpc.umpc_ref_nmpc[fe,self.umpc_key[u]] = self.reference_control_profile[u,self.nfe_t_0]
                    self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(max(abs(self.reference_control_profile[u,self.nfe_t_0]),1e-3))**2

    """ preparation and initialization """ 
    def set_predicted_state(self,m):
        self.predicted_state = {}
        for _var in m.component_objects(Var, active=True):
            for _key in _var.index_set():
                try:
                    self.predicted_state[(_var.name,_key)] = _var[_key].value
                except KeyError:
                    continue
                
    def cycle_ics(self,nmpc_as=False):
        for x in self.x:
            xic = getattr(self.olnmpc, x+'_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = self.curr_pstate[(x,j)] if nmpc_as else self.curr_rstate[(x,j)]
                

    def cycle_nmpc(self,**kwargs):
        m = self.olnmpc if self._c_it > 1 else self.recipe_optimization_model
        initialguess = kwargs.pop('init',self.store_results(m))
        # cut-off one finite element
        self.nfe_t -= 1
        
        # choose which type of nmpc controller is used
        if self.obj_type == 'economic':
            self.create_enmpc()
        else:
            self.create_nmpc()
        
        # initialize the new problem with shrunk horizon by the old one
        for _var in self.olnmpc.component_objects(Var, active=True):
            for _key in _var.index_set():
                if not(_key == None or type(_key) == str): # if the variable is time invariant scalar skip this
                    if (type(_key) == tuple and type(_key[0]) == int): # for variables that are indexed not only by number of finite element      
                        if _key[0] == self.min_horizon and self.nfe_t == self.min_horizon:
                            shifted_element = (_key[0],)
                        else:
                            shifted_element = (_key[0] + 1,)   # shift finite element by 1
                        aux_key = (_var.name,shifted_element + _key[1:]) # adjust key
                    elif type(_key) == int: # for variables that are only indexed by number of finite element
                        if _key == self.min_horizon and self.nfe_t == self.min_horizon:
                            shifted_element = _key
                        else:
                            shifted_element = _key + 1      # shift finite element by 1
                        aux_key = (_var.name,shifted_element)
                    else:
                        aux_key = (_var.name,_key)
                else:
                    aux_key = (_var.name,_key)
                try:
                    _var[_key] = initialguess[aux_key]
                except KeyError:
                    continue 
        
        # adapt parameters obtained from on-line estimation
        if self.adapt_params and self._c_it > 1:
            for index in self.curr_epars:
                p = getattr(self.olnmpc, index[0])
                key = index[1] if index[1] != () else None
                p[key].value = self.curr_epars[index]
        
        # set initial value parameters in model olnmpc
        # set according to predicted state by forward simulation
        # a) values will not be overwritten in case advanced step is used
        # b) values will be overwritten by call of add_noise() in case advanced step is not used
        for x in self.x:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = self.curr_pstate[(x,j)]
        
    """ create models """     
    def create_enmpc(self):
        self.olnmpc = self.d_mod(self.nfe_t,self.ncp_t)
        self.olnmpc.name = "olnmpc (Open-Loop eNMPC)"
        self.olnmpc.create_bounds()
        self.create_tf_bounds(self.olnmpc)
        self.olnmpc.clear_aux_bounds()
        
        self.olnmpc.K_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True)
        expression = 0.0
        if self.delta_u:
            for u in self.u:    
                control = getattr(self.olnmpc, u)
                for key in control.index_set():
                    if key > 1:
                        expression += self.olnmpc.K_w_nmpc[key]*(control[key-1] - control[key])**2.0
                    else:
                        expression += self.olnmpc.K_w_nmpc[1]*(self.nmpc_trajectory[self._c_it,u] - control[1])**2.0
        
        # generate the expressions for the objective function
        self.olnmpc.uK_expr_nmpc = Expression(expr = expression)
        self.olnmpc.eobj.expr += self.olnmpc.uK_expr_nmpc
        
    def create_nmpc(self):
        self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t)
            
        self.olnmpc.name = "olnmpc (Open-Loop NMPC)"
        self.olnmpc.create_bounds()
        self.create_tf_bounds(self.olnmpc) # creates tf_bounds
        self.olnmpc.clear_aux_bounds()
        if not(hasattr(self.olnmpc, 'ipopt_zL_in')):
            self.olnmpc.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            self.olnmpc.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            self.olnmpc.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            self.olnmpc.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            self.olnmpc.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
       
        # preparation for tracking objective function
        
        self.xmpc_l = {} # dictionary that includes a list of the state variables 
                         # for each finite element at cp = ncp_t 
                         # -> {finite_element:[list of states x(j)[finite_element, ncp]]} 
        
        self.xmpc_key = {} # encodes which state variable takes which index in the list stored in xmpc_l
        
        k = 0
        for t in range(1, self.nfe_t + 1):
            self.xmpc_l[t] = []
            for x in self.x:
                x_var = getattr(self.olnmpc, x)  #: State
                for j in self.x[x]:
                    self.xmpc_l[t].append(x_var[(t, self.ncp_t) + j])
                    if t == 1: # only relevant for the first run, afterwards repeating
                        self.xmpc_key[(x, j)] = k
                        k += 1
    
        # same for inputs
        self.umpc_l = {}
        self.umpc_key = {}
        k = 0
        for t in range(1, self.nfe_t + 1):
            self.umpc_l[t] = []
            for u in self.u:
                u_var = getattr(self.olnmpc, u)
                self.umpc_l[t].append(u_var[t])
                if t == 1: # only relevant for the first run, afterwards repeating
                    self.umpc_key[u] = k
                    k += 1
        
        # Parameter Sets that help to index the different states/controls for 
        # tracking terms according to xmpc_l/umpc_l + regularization for control steps delta_u
        self.olnmpc.xmpcS_nmpc = Set(initialize=[i for i in range(0, len(self.xmpc_l[1]))])
        self.olnmpc.umpcS_nmpc = Set(initialize=[i for i in range(0, len(self.umpc_l[1]))])
        # A: The reference trajectory
        self.olnmpc.xmpc_ref_nmpc = Param(self.olnmpc.fe_t, self.olnmpc.xmpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.umpc_ref_nmpc = Param(self.olnmpc.fe_t, self.olnmpc.umpcS_nmpc, initialize=0.0, mutable=True)
        # B: Weights for the different states (for x (Q) and u (R))
        self.olnmpc.Q_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=1.0, mutable=True)  
        self.olnmpc.R_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=1.0, mutable=True) 
        # C: Weights for the different finite elements (time dependence) (for x (Q) and u (R))
        self.olnmpc.Q_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True) 
        self.olnmpc.R_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True) 
        self.olnmpc.K_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True)
        
        # control step regularization
        expression = 0.0
        if self.delta_u:
            for u in self.u:    
                control = getattr(self.olnmpc, u)
                expression += self.olnmpc.K_w_nmpc[1]*(self.nmpc_trajectory[self._c_it,u] - control[1])**2.0 + sum(self.olnmpc.K_w_nmpc[fe]*(control[fe]-control[fe-1])**2.0 for fe in range(2,self.nfe_t + 1))
        
        # generate the expressions for the objective function
        self.olnmpc.uK_expr_nmpc = Expression(expr = expression)
        # A: sum over state tracking terms
        self.olnmpc.xQ_expr_nmpc = Expression(expr=sum(
        sum(self.olnmpc.Q_w_nmpc[fe] *
            self.olnmpc.Q_nmpc[k] * (self.xmpc_l[fe][k] - self.olnmpc.xmpc_ref_nmpc[fe,k])**2 for k in self.olnmpc.xmpcS_nmpc)
            for fe in range(1, self.nfe_t+1)))

        # B: sum over control tracking terms
        self.olnmpc.xR_expr_nmpc = Expression(expr=sum(
        sum(self.olnmpc.R_w_nmpc[fe] *
                self.olnmpc.R_nmpc[k] * (self.umpc_l[fe][k] - self.olnmpc.umpc_ref_nmpc[fe,k]) ** 2 for k in
                self.olnmpc.umpcS_nmpc) for fe in range(1, self.nfe_t + 1)))
        
        # deactive economic obj function (used in recipe optimization)
        self.olnmpc.eobj.deactivate()
        
        # declare/activate tracking obj function
        self.olnmpc.objfun_nmpc = Objective(expr = self.olnmpc.eobj.expr + self.olnmpc.xQ_expr_nmpc + self.olnmpc.xR_expr_nmpc + self.olnmpc.uK_expr_nmpc, sense=minimize)
                
    def create_noisy_model(self):
        self.noisy_model.tf = self.recipe_optimization_model.tf.value
        self.noisy_model.tf.fixed = True
        self.noisy_model.create_bounds()
        self.noisy_model.eobj.deactivate()
        # improve here and deactivate automatically
        self.noisy_model.epc_PO_ptg.deactivate()
        self.noisy_model.epc_unsat.deactivate()
        self.noisy_model.epc_PO_fed.deactivate()
        self.noisy_model.epc_mw.deactivate()
        
        # 1. remove initial conditions from the model
        k = 0
        for x in self.x:
            s = getattr(self.noisy_model, x)  #: state
            xicc = getattr(self.noisy_model, x + "_icc")
            xicc.deactivate()
            for j in self.x[x]:
                self.xp_l.append(s[(1, 0) + j])
                self.xp_key[(x, j)] = k
                k += 1
        
        # 2. introduce new variables for noise and reference noise
        self.noisy_model.xS_pnoisy = Set(initialize=[i for i in range(0, len(self.xp_l))])  #: Create set of noisy_states
        self.noisy_model.w_pnoisy = Var(self.noisy_model.xS_pnoisy, initialize=0.0)  #: Model disturbance
        self.noisy_model.w_ref = Param(self.noisy_model.xS_pnoisy,initialize=0.0, mutable=True)
        self.noisy_model.Q_pnoisy = Param(self.noisy_model.xS_pnoisy, initialize=1, mutable=True)
        # 3. redefine Objective: Find the noise that it is close to the randomly generated one but results in consistent initial conditions
        self.noisy_model.obj_fun_noisy = Objective(sense=minimize,
                                  expr=0.5 *
                                      sum(self.noisy_model.Q_pnoisy[k] * (self.noisy_model.w_pnoisy[k]-self.noisy_model.w_ref[k])**2 for k in self.noisy_model.xS_pnoisy))
    
        # 4. define new initial conditions + noise
        self.noisy_model.ics_noisy = ConstraintList()
        for x in self.x:
            s = getattr(self.noisy_model, x)  #: state
            xic = getattr(self.noisy_model, x + "_ic")
            for j in self.x[x]:
                xkey = None if j == () else j
                expr = s[(1, 0) + j] == xic[xkey] + self.noisy_model.w_pnoisy[self.xp_key[(x,j)]] # changed(!)
                self.noisy_model.ics_noisy.add(expr)
           
    
    """ optimization/simulation calls """                        
    def recipe_optimization(self,**kwargs):
        self.journalizer('O',self._c_it,'off-line recipe optimization','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        self.recipe_optimization_model = self.d_mod(self.nfe_t, self.ncp_t)
        self.recipe_optimization_model.initialize_element_by_element()
        self.recipe_optimization_model.create_bounds()
        self.recipe_optimization_model.clear_aux_bounds()
        self.recipe_optimization_model.create_output_relations()
        self.create_tf_bounds(self.recipe_optimization_model)

        if self.linapprox:
            cons = kwargs.pop('cons', [])
            eps = kwargs.pop('eps',0.0)
            iterlim = kwargs.pop('iterlim',10)
            self.solve_olrnmpc(model=self.recipe_optimization_model, iterlim=iterlim, cons=cons, eps=eps)
       
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["tol"] = 1e-8
        ip.options["linear_solver"] = "ma57"
        ip.options["max_iter"] = 5000
            
        results = ip.solve(self.recipe_optimization_model, tee=True, symbolic_solver_labels=True, report_timing=True)
        
        if not(str(results.solver.status) == 'ok' and str(results.solver.termination_condition)) == 'optimal':
            sys.exit('Error: Recipe optimization failed!')
            
        
        self.nmpc_trajectory[1,'tf'] = self.recipe_optimization_model.tf.value # start at 0.0
        for u in self.u:
            control = getattr(self.recipe_optimization_model,u)
            self.nmpc_trajectory[1,u] = control[1].value # control input bewtween time step i-1 and i
            self.curr_u[u] = control[1].value
        
        # directly apply state as predicted state --> forward simulation would reproduce the exact same result since no additional measurement is known
        for x in self.x:
            xvar = getattr(self.recipe_optimization_model,x)
            for j in self.x[x]:
                self.curr_pstate[(x,j)] = xvar[(1,self.ncp_t)+j].value
                self.initial_values[(x,j)] = xvar[(1,self.ncp_t)+j].value      
                
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    
    def get_tf(self):
        return [i*self.recipe_optimization_model.tf.value for i in range(1,self.nfe_t_0+1)]
            
    def solve_olnmpc(self):
        self.journalizer('U',self._c_it,'open-loop optimal control problem','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 1000
        with open("ipopt.opt", "w") as f:
            f.write("print_info_string yes")
        f.close()
        
        # enable l1-relaxation but set relaxation parameters to 0
        #self.olnmpc.eps.unfix()
        for i in self.olnmpc.eps.index_set():
            self.olnmpc.eps[i].value = 0 
            
        self.olnmpc.clear_aux_bounds() #redudant I believe
        
        results = ip.solve(self.olnmpc,tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.write_nl()
        if not(str(results.solver.status) == 'ok' and str(results.solver.termination_condition) == 'optimal'):
            print('olnmpc failed to converge')
            
        # save relevant results
        # self._c_it + 1 holds always if solve_olnmpc called at the correct time
        self.nmpc_trajectory[self._c_it,'solstat'] = [str(results.solver.status),str(results.solver.termination_condition)]
        self.nmpc_trajectory[self._c_it+1,'tf'] = self.nmpc_trajectory[self._c_it,'tf'] + self.olnmpc.tf.value
        self.nmpc_trajectory[self._c_it,'eps'] = [self.olnmpc.eps[i].value for i in self.olnmpc.eps.index_set()]  
        if self.obj_type == 'economic':
            self.nmpc_trajectory[self._c_it,'obj_value'] = value(self.olnmpc.eobj)
        else:
            self.nmpc_trajectory[self._c_it,'obj_value'] = value(self.olnmpc.objfun_nmpc)
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.nmpc_trajectory[self._c_it+1,u] = control[1].value # control input between timestep i-1 and i
        
        # initial state is saved to keep track of how good the estimates are
        # here only self.iteration since its the beginning of the olnmpc run
        for x in self.x:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                self.nmpc_trajectory[self._c_it,(x,j)] = xic[xkey].value   
                
        # save the control result as current control input
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.curr_u[u] = control[1].value
            
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
        
    def solve_olrnmpc(self,cons,**kwargs):
        self.journalizer('U', self._c_it, 'open-loop optimal control with sensitivity-based back-off margins','')
        # open-loop robust nonlinear model predictive control problem with linear approximation of the lower level program
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        
        # Set model
        m = kwargs.pop('model',self.olnmpc)
        
        # specify inputs
            # cons: list of the 'name's of the slack variables 's_name'
    
        # set algorithmic options
        iterlim = kwargs.pop('iterlim',10)  
        bt_iterlim = kwargs.pop('bt_iterlim',10)
        eps = kwargs.pop('eps',0.0)
        
        ip = SolverFactory('asl:ipopt')
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["max_iter"] = 5000
        ip.options["tol"] = 1e-8
        ip.options["linear_solver"] = "ma57"
        
        k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
        k_aug.options["compute_dsdp"] = ""
        k_aug.options["no_scale"] = ""
        iters = 0
        converged = False
        
        # initialize everything
        backoff = {}
        backoff_pre = {}
        for i in cons:
            backoff_var = getattr(m,'xi_'+i)
            for index in backoff_var.index_set():
                backoff[('s_'+i,index)] = backoff_pre[('s_'+i,index)] = 0.0
                backoff_var[index].value = 0.0
                
        n_p = 0
        if type(self.alpha) == float or (type(self.alpha) == tuple and self.alpha[1] == 'adapted'):
            # case a): hypercube or estimated parameters       
            for p in self.p_noisy:
                for key in self.p_noisy[p]:
                    n_p += 1
        elif type(self.alpha) == dict:
            # case b): hyperrectangle
            for p in self.alpha:
                n_p += 1

        print('iterations:')
        while (iters < iterlim and not(converged)):
            # solve optimization problem
            m.create_bounds()
            self.create_tf_bounds(m)
            m.clear_aux_bounds()
            for u in self.u:
                u_var = getattr(m,u)
                u_var.unfix()
            m.tf.unfix()
            m.eps.unfix()
            m.eps_pc.unfix()
            
            for i in cons:
                slack = getattr(m, 's_'+i)
                for index in slack.index_set():
                    slack[index].setlb(0.0)
            
            # check whether optimal control problem feasible
            # otw. backtrack back-off margin
            rho = 0.8
            b = 0
            infeas = True
            abort = max([np.floor(np.log(backoff_pre[i]/backoff[i])/np.log(rho))\
                         if backoff_pre[i] > 1e-5 else 0 for i in backoff]) if iters != 1 else bt_iterlim
            while(infeas and b < bt_iterlim):
                nlp_results = ip.solve(m, tee=False, symbolic_solver_labels=True)
                infeas = False
                if [str(nlp_results.solver.status),str(nlp_results.solver.termination_condition)] != ['ok','optimal']:
                    m.write_nl()
                    sys.exit('Error: Iteration in olrnmpc did not converge')
                for index in m.eps.index_set():
                    xikey = None if type(index) == int else index[:-1]
                    if m.eps[index].value > 1e-1:
                        cname = self.olnmpc.epc_indices[index]
                        xi = getattr(self.olnmpc, 'xi_' + cname)
                        xi[xikey] = rho*xi[xikey].value
                        backoff[('s_'+cname,xikey)] = xi[xikey].value
                        infeas = True
                        
                for index in m.eps_pc.index_set():
                    xikey = None if type(index) == int else index[:-1]
                    if m.eps_pc[index].value > 1e-1:
                        cname = self.olnmpc.pc_indices[index[-1]]
                        xi = getattr(self.olnmpc, 'xi_' + cname)
                        xi[xikey] = rho*xi[xikey].value
                        backoff[('s_'+cname,xikey)] = xi[xikey].value
                        infeas = True
                b += 1       
                        
                if infeas and b < min(bt_iterlim,abort):
                    print('Restricted Problem infeasible --> backtrack')
                    continue
            backoff_pre = deepcopy(backoff) 
            # break if OCP is infeasbile after the max number of backtracking 
            # iterations is reached
            if infeas and b == min(bt_iterlim,abort):
                break
            
            print('iteration ' + str(iters) + ' converged')
            
            m.eps.fix()
            m.eps_pc.fix()
            for u in self.u:
                u_var = getattr(m,u)
                u_var.fix()
            m.tf.fix()
            m.clear_all_bounds()
            
            for i in cons:
                slack = getattr(m, 's_'+i)
                for index in slack.index_set():
                    slack[index].setlb(None)
                
            for var in m.ipopt_zL_in:
                var.set_suffix_value(m.ipopt_zL_in, 0.0)
                
            for var in m.ipopt_zU_in:
                var.set_suffix_value(m.ipopt_zU_in, 0.0)
            # compute sensitivities
            if iters == 0:
                m.var_order = Suffix(direction=Suffix.EXPORT)
                m.dcdp = Suffix(direction=Suffix.EXPORT)
                i = 1
                reverse_dict_pars ={}
                
                if type(self.alpha) == float or (type(self.alpha) == tuple and self.alpha[1] == 'adapted'):
                    # case a): hypercube or estimated parameters
                    for p in self.p_noisy:
                        for key in self.p_noisy[p]:
                            if key == ():
                                dummy = 'dummy_constraint_r_' + p 
                            else:
                                dummy = 'dummy_constraint_r_' + p + '_' + str(key[0])
                            dummy_con = getattr(m, dummy)
                            for index in dummy_con.index_set():
                                m.dcdp.set_value(dummy_con[index], i)
                                reverse_dict_pars[i] = (p,key)
                                i += 1
                elif type(self.alpha) == dict:
                    # case b): hyperrectangle
                    for par in self.alpha:
                        p = par[0]
                        key = par[1]
                        if key == ():
                            dummy = 'dummy_constraint_r_' + p 
                        else:
                            dummy = 'dummy_constraint_r_' + p + '_' + str(key[0])
                        dummy_con = getattr(m, dummy)
                        for index in dummy_con.index_set():
                            m.dcdp.set_value(dummy_con[index], i)
                            reverse_dict_pars[i] = (p,key)
                            i += 1                  
                i = 1
                reverse_dict_cons = {}
                for k in cons:
                    s = getattr(m, 's_'+k)
                    for index in s.index_set():
                        if not(s[index].stale):
                            m.var_order.set_value(s[index], i)
                            reverse_dict_cons[i] = ('s_'+ k,index)
                            i += 1
                
            #compute sensitivities
            k_aug.solve(m, tee=False)
            m.write_nl()
            sens = {}
            
            with open('dxdp_.dat') as f:
                reader = csv.reader(f, delimiter="\t")
                i = 1
                for row in reader:
                    k = 1
                    s = reverse_dict_cons[i]
                    for col in row[1:]:
                        p = reverse_dict_pars[k]
                        sens[(s,p)] = float(col)
                        k += 1
                    i += 1
            # compute weighted sensitivity matrix
            if type(self.alpha) == float:
                # case a): hypercube
                for key in sens:
                    sens[key] *= self.alpha
            elif type(self.alpha) == dict:
                # case b): hyperrectangle can also be adapted
                for key in sens:
                    sens[key] *= self.alpha[key[1]]
                """
            NOT READY FOR USE
            elif type(self.alpha) == tuple and self.alpha[1] == 'adapted':
                # case c): weighting_matrix --> the enclosing 1-norm is tilted 
                # it would be smarter to use the bounds obtained from the
                # principle components of the confidence ellipsoid and just tilt
                # it according to U in A = USV'
                
                # self made matrix multiplication to ensure the rows and cols match...
                
                if type(self._scaled_shape_matrix) != dict:
                    for key in sens:
                        #key[1] = p # the exact same index for self.PI_indices
                        #self.PI_indices[key[1]] returns index for corresponding parameter in _PI
                        s = key[0] 
                        if key[1] in self.PI_indices:
                            sens[key] = sum(sens[s,p]*self._scaled_shape_matrix[self.PI_indices[key[1]]][self.PI_indices[p]] for p in self.PI_indices)
                        else:
                            sens[key] *= self.alpha[0][key[1]]# in case there are additional params that are not estimated
                else:
                    for key in sens:
                        sens[key] *= self.alpha[0][key[1]]
                """
            else:
                sys.exit('Error: Specification of uncertainty set not supported.')
            
            # convergence check and update  
            converged = True
            for i in cons:
                backoff_var = getattr(m,'xi_'+i)
                slack_var = getattr(m,'s_'+i)
                for index in backoff_var.index_set():
                    try:
                        # computes the 1-norm of the respective part of the sensitivity matrix
                        # ToDo: can be extended to computing the 2-norm --> ellipsoidal uncertainty set
                        new_backoff = sum(abs(sens[(('s_'+i,index),reverse_dict_pars[k])]) for k in range(1,n_p+1))
                        # update backoff margins
                        
                        if backoff[('s_'+i,index)] - new_backoff <= -eps - slack_var[index].value:
                            backoff[('s_'+i,index)] = new_backoff
                            backoff_var[index].value = new_backoff
                            converged = False
                            # check convergence
                            #if  old_backoff - new_backoff <= -eps:
                            #    converged = False
                        else:
                            continue
                    except KeyError:
                        # catches all the stale/redundant slacks
                        continue
            iters += 1
            
        if m == self.olnmpc:    
            m.create_bounds()
            self.create_tf_bounds(m)
            m.clear_aux_bounds()
            for u in self.u:
                u_var = getattr(m,u)
                u_var.unfix()
            m.tf.unfix()
            m.eps.unfix()
            m.eps_pc.unfix()
            for i in cons:
                slack = getattr(m, 's_'+i)
                for index in slack.index_set():
                    slack[index].setlb(0)
            nlp_results = ip.solve(m,tee=True, symbolic_solver_labels=True)
            # save converged results 
            self.nmpc_trajectory[self._c_it,'solstat'] = [str(nlp_results.solver.status),str(nlp_results.solver.termination_condition)]       
            self.nmpc_trajectory[self._c_it+1,'tf'] = self.nmpc_trajectory[self._c_it,'tf'] + m.tf.value
            self.nmpc_trajectory[self._c_it,'eps'] = [m.eps[i].value for i in m.eps.index_set()]  
            
            if self.obj_type == 'economic':
                self.nmpc_trajectory[self._c_it,'obj_value'] = value(m.eobj)
            else:
                self.nmpc_trajectory[self._c_it,'obj_value'] = value(m.objfun_nmpc)
            
            for u in self.u:
                control = getattr(m,u)
                self.nmpc_trajectory[self._c_it+1,u] = control[1].value # control input between timestep i-1 and i
            
            for x in self.x:
                xic = getattr(m, x + '_ic')
                for j in self.x[x]:
                    xkey = None if j == () else j
                    self.nmpc_trajectory[self._c_it,(x,j)] = xic[xkey].value   

            
            # save the control result as current control input
            for u in self.u:
                control = getattr(m,u)
                self.curr_u[u] = control[1].value
            
        
            m.write_nl()
        else:
            pass    
        
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    
    def add_noise(self,var_dict):
        self.journalizer('Pl', 'auxilliary optimization problem for consistent process noise','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        # set the time horizon
        self.noisy_model.tf = self.recipe_optimization_model.tf.value # just a dummy value in order to prohibit tf to go to 0
        # to account for possibly inconsistent initial values
        # solve auxilliary problems with only 1 finite element  
        for x in self.x:
            xic = getattr(self.noisy_model, x + '_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = self.curr_rstate[(x,j)] 
        
        # draw random numbers from normal distribution and compute absolut! noise
        noise_init = {}
        for key in var_dict:
            noise_init[key] = np.random.normal(loc=0.0, scale=var_dict[key])
            xic = getattr(self.noisy_model, key[0] + '_ic')
            if len(key[1]) > 0:
                noise_init[key] = noise_init[key]*xic[key[1:]].value
            else:
                noise_init[key] = noise_init[key]*xic.value
        # 5. define the weighting factor based on the standard deviation
        for key in var_dict:
            v_i = self.xp_key[key]
            xic = getattr(self.noisy_model, key[0] + '_ic')
            try:
                if len(key[1]) > 0:
                    self.noisy_model.Q_pnoisy[v_i].value = 1.0/(var_dict[key]*xic[key[1:]].value + 0.01)
                else:
                    self.noisy_model.Q_pnoisy[v_i].value = 1.0/(var_dict[key]*xic.value + 0.01)
            except ZeroDivisionError:
                self.noisy_model.Q_pnoisy[v_i].value = 1.0
        # 6. define the reference noise values w_ref as randomly generated ones and set the initial guess to the same value
            self.noisy_model.w_ref[v_i].value = noise_init[key]
            self.noisy_model.w_pnoisy[v_i].value = noise_init[key]
            if var_dict[key] == 0:
                self.noisy_model.w_pnoisy[v_i].fixed = True
            
        
        # 7. solve the problem
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8

        results = ip.solve(self.noisy_model, tee=True, symbolic_solver_labels=True)
        
        self.nmpc_trajectory[self._c_it, 'solstat_noisy'] = [str(results.solver.status),str(results.solver.termination_condition)]
        self.nmpc_trajectory[self._c_it, 'obj_noisy'] = value(self.noisy_model.obj_fun_noisy.expr)
        # save the new consistent initial conditions 
        for key in var_dict:
            vni = self.xp_key[key]
            self.curr_rstate[key] = self.noisy_model.w_pnoisy[vni].value + self.curr_rstate[key] # from here on it is a noisy real state
            
        # Set new (noisy) initial conditions in model self.olnmpc
        # + save in real_trajectory dictionary
        for x in self.x:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = self.curr_rstate[(x,j)]                   
              
        # for trouble shooting to check how much the consistent initial conditions deviates from the completely random one
        for key in var_dict:    
            vni = self.xp_key[key]
            self.nmpc_trajectory[self._c_it,'noise',key] = self.noisy_model.w_pnoisy[vni].value - self.noisy_model.w_ref[vni].value
        
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)

    def plant_simulation(self, **kwargs):
        self.journalizer('Pl',self._c_it,'plant simulation','')   
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        
        m = self.olnmpc if self._c_it > 1 else self.recipe_optimization_model
        initialguess = kwargs.pop('init',self.store_results(m))
        
        # simulates the current 
        disturbance_src = kwargs.pop('disturbance_src', 'process_noise')
        # options: process_noise --> gaussian noise added to all states
        #          parameter noise --> noise added to the uncertain model parameters
        #                              or different ones
        #          input noise --> noise added to the inputs/controls
        #          parameter_scenario --> prespecified parameter values (tailored to time-invariant)
        # not exhaustive list, can be adapted/extended as one wishes
        # combination of all the above with noise in the initial point supported
        initial_disturbance = kwargs.pop('initial_disturbance', {(x,j):0.0 for x in self.x for j in self.x[x]})
        parameter_disturbance = kwargs.pop('parameter_disturbance', {})
        state_disturbance = kwargs.pop('state_disturbance', {})
        input_disturbance = kwargs.pop('input_disturbance', {})
        parameter_scenario = kwargs.pop('scenario', {})        
        
        #  generate the disturbance according to specified scenario
        state_noise = {}
        input_noise = {}
        if disturbance_src == 'process_noise':
            if state_disturbance != {}: 
                for key in state_disturbance:
                    state_noise[key] = np.random.normal(loc=0.0, scale=state_disturbance[key])
                    # potentially implement truncation at 2 sigma
            else:
                for x in self.x: 
                    for j in self.x[x]:
                        state_noise[(x,j)] = 0.0   
            for u in self.u:
                input_noise[u] = 0.0     
        elif disturbance_src == 'parameter_noise':
            for p in parameter_disturbance:
                disturbed_parameter = getattr(self.plant_simulation_model, p[0])
                pkey = None if p[1] == () else p[1]
                if self._c_it == 1:  
                    self.nominal_parameter_values[p] = deepcopy(disturbed_parameter[pkey].value)
                if (self._c_it-1)%parameter_disturbance[p][1] == 0:
                    sigma = parameter_disturbance[p][0]
                    rand = np.random.normal(loc=0.0, scale=sigma)
                    #truncation at 2 sigma
                    if abs(rand) > 2*sigma:
                        rand = -2.0 * sigma if sigma < 0.0 else 2.0 * sigma
                    disturbed_parameter[pkey].value = self.nominal_parameter_values[p] * (1 + rand)                 
            for x in self.x:
                for j in self.x[x]:
                    state_noise[(x,j)] = 0.0
                    
            for u in self.u:
                input_noise[u] = 0.0
        elif disturbance_src == 'input_disturbance': # input noise only
            for x in self.x:
                for j in self.x[x]:
                    state_noise[(x,j)] = 0.0
            for u in self.u:
                input_noise[u] = np.random.normal(loc=0.0, scale=input_disturbance[u])
        elif disturbance_src == 'parameter_scenario':
            for p in parameter_scenario:
                disturbed_parameter = getattr(self.plant_simulation_model, p[0])
                pkey = None if p[1] == () else p[1]
                if self._c_it == 1:
                    self.nominal_parameter_values[p] = deepcopy(disturbed_parameter[pkey].value)
                disturbed_parameter[pkey].value = self.nominal_parameter_values[p]*(1 + parameter_scenario[p])
                        
            for x in self.x:
                for j in self.x[x]:
                    state_noise[(x,j)] = 0.0
                        
            for u in self.u:
                input_noise[u] = 0.0
        else:
            print('NO DISTURBANCE SCENARIO SPECIFIED, NO NOISE ADDED ANYWHERE')                     
            for x in self.x:
                for j in self.x[x]:
                        state_noise[(x,j)] = 0.0
            for u in self.u:
                input_noise[u] = 0.0     
        if self._c_it == 1: 
            for x in self.x:
                xic = getattr(self.plant_simulation_model,x+'_ic')
                x_var = getattr(self.plant_simulation_model,x)
                for j in self.x[x]:
                    xkey = None if j == () else j
                    xic[xkey].value = xic[xkey].value * (1 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j)]))
            
            # initialization of the simulation (initial guess)               
            for var in self.plant_simulation_model.component_objects(Var, active=True):
                var_ref = getattr(self.recipe_optimization_model, var.name)
                for key in var.index_set():
                    if var[key].fixed:
                        break
                    else:
                        var[key].value = var_ref[key].value
        else:                
            # initialization of state trajectories
            # adding state noise if specified
            for x in self.x:
                xic = getattr(self.plant_simulation_model,x+'_ic')
                x_var = getattr(self.plant_simulation_model,x)
                for j in self.x[x]:
                    xkey = None if j == () else j
                    xic[xkey].value = x_var[(1,self.ncp_t)+j].value * (1.0 + state_noise[(x,j)])# + noise # again tailored to RADAU nodes
 
            # initialization of simulation
            for var in self.plant_simulation_model.component_objects(Var, active=True):
                try:
                    var_ref = getattr(self.olnmpc,var.name)
                except AttributeError: 
                    # catch that plant simulation includes quantities (Outputs)
                    # that are not relevant for optimal control problem (therefore removed)
                    continue
                for key in var.index_set():
                    if var[key].fixed:
                        break
                    else:
                        var[key].value = var_ref[key].value
           
        # result is the previous olnmpc solution 
        #    --> therefore provides information about the sampling interval
        # 1 element model for forward simulation
        self.plant_simulation_model.tf = initialguess['tf', None]
        self.plant_simulation_model.tf.fixed = True
        
        # FIX(!!) the controls, path constraints are deactivated by default
        for u in self.u:
            control = getattr(self.plant_simulation_model,u)
            control[1].fix(initialguess[u,1]*(1.0+input_noise[u]))
        
        self.plant_simulation_model.equalize_u(direction="u_to_r")
        self.plant_simulation_model.clear_aux_bounds()

        # solve statement
        ip = SolverFactory("asl:ipopt")
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 1000
        
        
        self.plant_simulation_model.clear_all_bounds()
        out = ip.solve(self.plant_simulation_model, tee=True, symbolic_solver_labels=True)
        
        # check if converged otw. run again and hope for numerical issues
        if [str(out.solver.status), str(out.solver.termination_condition)] != ['ok','optimal']:
            self.plant_simulation_model.create_bounds()
            self.plant_simulation_model.clear_aux_bounds()
            out = ip.solve(self.plant_simulation_model, tee = True, symbolic_solver_labels=True)
            
        self.plant_trajectory[self._c_it,'solstat'] = [str(out.solver.status), str(out.solver.termination_condition)]
        self.plant_trajectory[self._c_it,'tf'] = self.plant_simulation_model.tf.value
        
        # safe results of the plant_trajectory dictionary {number of iteration, (x,j): value}
        for u in self.u:
            control = getattr(self.plant_simulation_model,u)
            self.plant_trajectory[self._c_it,u] = control[1].value #(control valid between 0 and tf)
        for x in self.x:
            xvar = getattr(self.plant_simulation_model, x)
            for j in self.x[x]:
                    self.plant_trajectory[self._c_it,(x,j)] = xvar[(1,3)+j].value 
                    # setting the current real value w/o measurement noise
                    self.curr_rstate[(x,j)] = xvar[(1,self.ncp_t)+j].value 
      
        # monitor path constraints for results
        for pc in self.path_constraints:
            pc_var = getattr(self.plant_simulation_model, pc)
            for index in pc_var.index_set():
                self.pc_trajectory[(pc,(self._c_it,index[1:]))] = pc_var[index].value
        
        # monitor other interesting properties
        self.monitor[self._c_it] = {}
        for poi in self.poi:
            poi_var = getattr(self.plant_simulation_model, poi)
            for index in poi_var.index_set():
                self.monitor[self._c_it][(poi,index)] = poi_var[index].value
                
        for cp in range(self.ncp_t+1):
            self.pc_trajectory[('tf',(self._c_it,cp))] = self.plant_simulation_model.tau_i_t[cp]*self.plant_simulation_model.tf.value

        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)

    def forward_simulation(self):
        self.journalizer('P',self._c_it,'forward simulation','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        # simulates forward from the current measured or (!) esimated state
        # using the sensitivity based updated control inputs
        
        # IMPORTANT:
        # --> before calling forward_simulation() need to call dot_sens to perform the update
                
        # curr_state_info:
        # comprises both cases where state is estimated and directly measured
        current_state_info = {}
        for key in self.curr_pstate:
            current_state_info[key] = self.curr_pstate[key] - self.curr_state_offset[key]
           
        # save updated controls to current_control_info
        # apply clamping strategy if controls violate physical bounds
        current_control_info = {}
        for u in self.u:
            control = getattr(self.olnmpc, u)
            current_control_info[u] = control[1].value
          
        # change tf via as as well?
        self.forward_simulation_model.tf.fix(self.olnmpc.tf.value)
        
        # Update adpated params
        if self.adapt_params and self._c_it > 1:
            for index in self.curr_epars:
                key = index[1] if index[1] != () else None
                p = getattr(self.forward_simulation_model, index[0])
                p[key].value = self.curr_epars[index]

        # general initialization
        for var in self.olnmpc.component_objects(Var, active=True):
            xvar = getattr(self.forward_simulation_model, var.name)
            for key in var.index_set():
                try:
                    if key == None or type(key) == int:
                        xvar[key].value = var[key].value
                    elif type(key) == tuple:
                        if len(key) > 2:
                            xvar[key].value = var[(1,self.ncp_t)+key[2:]].value
                        else:
                            xvar[key].value = var[(1,self.ncp_t)].value
                except KeyError:
                    # catches exceptions for trying to access nfe>1 for self.forward_simulation_model
                    continue 
                    
        # 1 element model for forward simulation
        # set initial point (measured or estimated)
        #       --> implicit assumption: RADAU nodes
        for x in self.x:
            xic = getattr(self.forward_simulation_model,x+'_ic')
            xvar = getattr(self.forward_simulation_model,x)
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = current_state_info[(x,j)]
                # just initialization heuristic --> change if you want to
                for k in range(0,self.ncp_t+1):# use only 
                    xvar[(1,k)+j].value = current_state_info[(x,j)]
                    
        # set and fix control as provided by olnmpc/advanced step update     
        for u in self.u:
            control = getattr(self.forward_simulation_model,u)
            control[1].fix(current_control_info[u])           
            self.forward_simulation_model.equalize_u(direction="u_to_r")
        
        self.forward_simulation_model.clear_aux_bounds()
        # solve statement
        ip = SolverFactory("asl:ipopt")
        #ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 1000
        
        self.forward_simulation_model.clear_all_bounds()
        out = ip.solve(self.forward_simulation_model, tee=True, symbolic_solver_labels=True)
        self.simulation_trajectory[self._c_it,'obj_fun'] = 0.0
        
        if [str(out.solver.status), str(out.solver.termination_condition)] != ['ok','optimal']:
            self.forward_simulation_model.clear_all_bounds()
            out = ip.solve(self.forward_simulation_model, tee = True, symbolic_solver_labels=True)
            
        self.simulation_trajectory[self._c_it,'solstat'] = [str(out.solver.status), str(out.solver.termination_condition)]
        
        # save the simulated state as current predicted state
        # implicit assumption of RADAU nodes
        for x in self.x:
            xvar = getattr(self.forward_simulation_model, x)
            for j in self.x[x]:
                    self.curr_pstate[(x,j)] = xvar[(1,self.ncp_t)+j].value  

        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    
    """ open-loop considerations """
    def open_loop_simulation(self, sample_size = 10, **kwargs):
        # simulates the current
        # options: process_noise --> gaussian noise added to all states
        #          parameter noise --> noise added to the uncertain model parameters
        #                               or different ones
        #          input noise --> noise added to the inputs/controls
        # not exhaustive list, can be adapted/extended as one wishes
        # combination of all the above with noise in the initial point supporte
        initial_disturbance = kwargs.pop('initial_disturbance', {(x,j):0.0 for x in self.x for j in self.x[x]})
        parameter_disturbance = kwargs.pop('parameter_disturbance', {})
        input_disturbance = kwargs.pop('input_disturbance',{})
        parameter_scenario = kwargs.pop('parameter_scenario',{})
        
        # deactivate constraints
        self.recipe_optimization_model.deactivate_epc()
        self.recipe_optimization_model.deactivate_pc()
        self.recipe_optimization_model.eobj.deactivate()
        self.recipe_optimization_model.del_pc_bounds()
        self.recipe_optimization_model.clear_aux_bounds()
        
        nominal_initial_point = {}
        for x in self.x: 
            xic = getattr(self.recipe_optimization_model, x+'_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                nominal_initial_point[(x,j)] = xic[xkey].value
                    
        if parameter_disturbance != {} or parameter_scenario != {}:
            par_dict = parameter_disturbance if parameter_disturbance != {} else parameter_scenario[0]
            for p in par_dict:
                key = p[1] if p[1] != () else None
                disturbed_parameter = getattr(self.recipe_optimization_model, p[0])
                self.nominal_parameter_values[p] = disturbed_parameter[key].value
                    
        for u in self.u:
            control = getattr(self.recipe_optimization_model, u)
            control.fix()
            
        endpoint_constraints = {}
        pc_trajectory = {}
        # set and fix controls + add noise
        for k in range(sample_size):
            self.journalizer('O',k,'off-line optimal control','')
            pc_trajectory[k] = {}
            if initial_disturbance != {}:
                for x in self.x:
                    xic = getattr(self.recipe_optimization_model,x+'_ic')
                    for j in self.x[x]:
                        key = j if j != () else None
                        xic[key].value = nominal_initial_point[(x,j)] * (1 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j)]))
            if input_disturbance != {}:
                for u in input_disturbance:
                    control = getattr(self.recipe_optimization_model, u)
                    for i in range(1,self.nfe_t+1):
                        disturbance_noise = np.random.normal(loc=0.0, scale=input_disturbance[u])
                        control[i,1].value = self.reference_control_profile[u,(i,1)]*(1+disturbance_noise)
            if parameter_disturbance != {}:
                for p in parameter_disturbance:
                    key = p[1] if p[1] != () else None
                    disturbed_parameter = getattr(self.recipe_optimization_model, p[0])
                    disturbed_parameter[key].value = self.nominal_parameter_values[p] * (1 + np.random.normal(loc=0.0, scale=parameter_disturbance[p][0]))    
            if parameter_scenario != {}:
                for p in parameter_scenario[k]:
                    key = p[1] if p[1] != () else None
                    disturbed_parameter = getattr(self.recipe_optimization_model, p[0])
                    disturbed_parameter[key].value = (1.0 + parameter_scenario[k][p])*self.nominal_parameter_values[p] 
            self.recipe_optimization_model.tf.fix()
            self.recipe_optimization_model.equalize_u(direction="u_to_r")
            # run the simulation
            ip = SolverFactory("asl:ipopt")
            ip.options["halt_on_ampl_error"] = "yes"
            ip.options["print_user_options"] = "yes"
            ip.options["linear_solver"] = "ma57"
            ip.options["tol"] = 1e-8
            ip.options["max_iter"] = 1000

            out = ip.solve(self.recipe_optimization_model, tee=True, symbolic_solver_labels=True)
            if  [str(out.solver.status), str(out.solver.termination_condition)] == ['ok','optimal']:
                converged = True
            else:
                converged = False
        
            if converged:
                endpoint_constraints[k] = self.recipe_optimization_model.check_feasibility(display=True)
                for pc_name in self.path_constraints:
                    pc_var = getattr(self.recipe_optimization_model, pc_name)
                    for fe in self.recipe_optimization_model.fe_t:
                        for cp in self.recipe_optimization_model.cp:
                            pc_trajectory[k][(pc_name,(fe,(cp,)))] = pc_var[fe,cp].value
                            pc_trajectory[k][('tf',(fe,cp))] = self.recipe_optimization_model.tau_i_t[cp]*self.recipe_optimization_model.tf.value
            else:
                sys.exit('Error: Simulation not converged!')
                endpoint_constraints[k] = 'error'
        
        return endpoint_constraints, pc_trajectory    
   

        
    """ sensitivity-based updates """
    def create_suffixes_nmpc(self, first_iter = False):
        """Creates the required suffixes for the olnmpc problem"""
        if first_iter:
            m = self.recipe_optimization_model
        else:
            m = self.olnmpc
            
        if hasattr(m, "npdp"):
            pass
        else:
            m.npdp = Suffix(direction=Suffix.EXPORT)
            
        if hasattr(m, "dof_v"):
            pass
        else:
            m.dof_v = Suffix(direction=Suffix.EXPORT)

        for u in self.u:
            uv = getattr(m, u)
            uv[1].set_suffix_value(m.dof_v, 1)

    def compute_offset_state(self, src_kind="estimated"):
        """Missing noisy"""
        if src_kind == "estimated":
            for x in self.x:
                for j in self.x[x]:
                    self.curr_state_offset[(x, j)] = self.curr_pstate[(x, j)] - self.curr_estate[(x, j)]
        elif src_kind == "real":
            for x in self.x:
                for j in self.x[x]:
                    self.curr_state_offset[(x, j)] = self.curr_pstate[(x, j)] - self.curr_rstate[(x, j)]
        
        # can be removed if I am done with troubleshooting
        self.nmpc_trajectory[self._c_it,'state_offset'] = self.curr_state_offset
             
    def sens_dot_nmpc(self):
        self.journalizer("I", self._c_it, "sens_dot_nmpc", "Set-up")
        if hasattr(self.olnmpc, "npdp"):
            self.olnmpc.npdp.clear()
        else:
            self.olnmpc.npdp = Suffix(direction=Suffix.EXPORT)

        for x in self.x:
            con_name = x + "_icc"
            con_ = getattr(self.olnmpc, con_name)
            for j in self.x[x]:
                xkey = None if j == () else j
                con_[xkey].set_suffix_value(self.olnmpc.npdp, self.curr_state_offset[(x, j)])


        if hasattr(self.olnmpc, "f_timestamp"):
            self.olnmpc.f_timestamp.clear()
        else:
            self.olnmpc.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        self.olnmpc.set_suffix_value(self.olnmpc.f_timestamp, self.int_file_nmpc_suf)

        self.olnmpc.f_timestamp.display(ostream=sys.stderr)

        self.journalizer("I", self._c_it, "sens_dot_nmpc", self.olnmpc.name)

        #before_dict = {}
        for u in self.u:
            control = getattr(self.olnmpc,u)
            #before_dict[u] = control[1].value

        results = self.dot_driver.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)
        
        #after_dict = {}
        #difference_dict = {}
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            #after_dict[u] = control[1].value
            #difference_dict[u] = after_dict[u] - before_dict[u]
                        
        ftiming = open("timings_dot_driver.txt", "r")
        s = ftiming.readline()
        ftiming.close()
        k = s.split()
        self._dot_timing = k[0]
        
        #flho: augmented
           
        # save updated controls to current_control_info
        # apply clamping strategy if controls violate bounds
        for u in self.u:
            control = getattr(self.olnmpc, u)
            if control[1].value > control[1].ub:
                if control[1].ub == None: # number > None == True 
                    pass
                else:
                    control[1] = control[1].ub
            elif control[1].value < control[1].lb: #  number < None == False
                control[1].value = control[1].lb
            else:
                pass
    
        #applied = {}
        for u in self.u:
            control = getattr(self.olnmpc,u)
            #applied[u] = control[1].value
            self.curr_u[u] = control[1].value
            
        #return before_dict, after_dict, difference_dict, applied
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
            
    def sens_k_aug_nmpc(self):
        self.journalizer("I", self._c_it, "sens_k_aug_nmpc", "k_aug sensitivity")
        self.olnmpc.ipopt_zL_in.update(self.olnmpc.ipopt_zL_out)
        self.olnmpc.ipopt_zU_in.update(self.olnmpc.ipopt_zU_out)
        self.journalizer("I", self._c_it, "solve_k_aug_nmpc", self.olnmpc.name)
        
        if hasattr(self.olnmpc, "f_timestamp"):
            self.olnmpc.f_timestamp.clear()
        else:
            self.olnmpc.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                             datatype=Suffix.INT)

        self.olnmpc.set_suffix_value(self.olnmpc.f_timestamp, self.int_file_nmpc_suf)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)
        #self.k_aug_sens.options["no_inertia"] = ""
        #self.k_aug_sens.options["no_barrier"] = ""
        #self.k_aug_sens.options['target_log10mu'] = -5.7
        self.k_aug_sens.options['no_scale'] = ""
        
        results = self.k_aug_sens.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)
        ftimings = open("timings_k_aug.txt", "r")
        s = ftimings.readline()
        ftimings.close()
        self._k_timing = s.split()
        
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
        
    def run(self,regularization_weights={},disturbance_src={},\
            olrnmpc_args={}, advanced_step=False):
        ru = {}
        # off-line open-loop control
        ru['recipe_optimization'] = self.recipe_optimization(**olrnmpc_args)
        self.set_reference_state_trajectory(self.get_state_trajectory(self.recipe_optimization_model))
        self.set_reference_control_profile(self.get_control_profile(self.recipe_optimization_model))
        
        self.create_nmpc() if self.obj_type == 'tracking' else self.create_enmpc() 
        self.load_reference_trajectories() if self.obj_type == 'tracking' else None
    
        # on-line control
        if advanced_step:
            for i in range(1,self.nfe_t_0):
                ru['plant_simulation',i]=self.plant_simulation(**disturbance_src)
                # preparation phase
                self.cycle_nmpc() 
                self.cycle_ics(as_nmpc=True)
                self.load_reference_trajectories() if self.obj_type == 'tracking' else None # loads the reference trajectory in olnmpc problem (for regularization)
                self.set_regularization_weights(**regularization_weights)
                ru['olnmpc',i]=self.solve_olrnmpc(**olrnmpc_args) if self.linapprox else self.solve_olnmpc() # solves the olnmpc problem
                self.create_suffixes_nmpc()
                ru['sens',i]=self.sens_k_aug_nmpc()
                #updating phase
                self.compute_offset_state(src_kind="real")
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
                self.cycle_nmpc()
                self.cycle_ics()
                # updating phase
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
        ru['plant_simulation',i] = self.plant_simulation(**disturbance_src)
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

class msNMPCGen(DynGen):
    def __init__(self, **kwargs):
        DynGen.__init__(self, **kwargs)
        self.int_file_nmpc_suf = int(time.time())+1

        # monitoring
        self.path_constraints = kwargs.pop('path_constraints',[])
        
        # regularization
        self.delta_u = kwargs.pop('control_displacement_penalty', False)
        
        # multistage
        dummy_tree = {}
        for i in range(1,self.nfe_t+1):
            dummy_tree[i,1] = (i-1,1,1,{})
        self.st = kwargs.pop('scenario_tree', dummy_tree)
        self.s_max = kwargs.pop('s_max', 1) # number of scenarios
        self.s_used = self.s_max
        self.nr = kwargs.pop('robust_horizon', 1) # robust horizon
        
        # objective type
        self.obj_type = kwargs.pop('obj_type','tracking')
        self.min_horizon = kwargs.pop('min_horizon',0)

        self.noisy_model = self.d_mod(1,self.ncp_t)
        self.recipe_optimization_model = object() 
        # model to compute the reference trajectory (open loop optimal control problem with economic obj)
        self.reference_state_trajectory = {} # {('state',(fe,j)):value)}
        self.reference_control_profile = {} # {('control',(fe,j)):value)}
        self.storage_model = object()
        self.initial_values = {}
        self.nominal_parameter_values = {}
        self.uncertainty_set = kwargs.pop('uncertainty_set',{})
        
        # open-loop control problem solved in every NMPC iteration
        self.olnmpc = object()
        
        # for advanced step: 
        self.forward_simulation_model = self.d_mod(1,self.ncp_t) # only need 1 element for forward simulation
        self.forward_simulation_model.name = 'forward simulation model'
        self.forward_simulation_model.create_output_relations()
        self.forward_simulation_model.create_bounds()
        self.forward_simulation_model.deactivate_pc()
        self.forward_simulation_model.deactivate_epc()
        self.forward_simulation_model.eobj.deactivate()
        self.simulation_trajectory = {}
        # organized as follows   
        #- for states:   {i, ('name',(index)): value} # actual state at time step i
        #- for controls: {i, 'name': value} 
                # --> control input valid between iteration i-1 and i 
                #     (i=1 --> is the first piecewise constant control input) 
        
        
        # plant simulation model in order to distinguish between noise and disturbances
        self.plant_simulation_model = self.d_mod(1, self.ncp_t)
        self.plant_simulation_model.name = 'plant simulation model'
        self.plant_simulation_model.create_output_relations()
        self.plant_simulation_model.create_bounds()
        self.plant_simulation_model.deactivate_pc()
        self.plant_simulation_model.deactivate_epc()
        self.plant_simulation_model.eobj.deactivate()
        self.plant_trajectory = {} 
        self.plant_trajectory[0,'tf'] = 0
        # organized as follows   
        #- for states:   {i, ('name',(index)): value} # actual state at time step i
        #- for controls: {i, 'name': value} 
                # --> control input valid between iteration i-1 and i 
                #     (i=1 --> is the first piecewise constant control input) 
        
        self.nmpc_trajectory = {} 
        self.nmpc_trajectory[0,'tf'] = 0 # initialize final-time computation
        #- for states:   {i, ('name',(index)): value} # actual state at time step i
        #- for controls: {i, 'name': value} 
                # --> control input valid between iteration i-1 and i 
                #     (i=1 --> is the first piecewise constant control input) 
                
        self.pc_trajectory = {} # structure: {(pc,(iteration,collocation_point):value}
    
        self.poi = kwargs.pop('poi',[])
        self.monitor = {}           
    
    """ setting reference trajectories for tracking-type objectives """
    def set_reference_state_trajectory(self,input_trajectory):
        # input_trajectory = {('state',(fe,cp,j)):value} ---> this format
        self.reference_state_trajectory = input_trajectory

    def get_state_trajectory(self,d_mod):
        output = {}
        for state_name in self.x:
            xvar = getattr(d_mod,state_name)
            for key in xvar.index_set():
                if key[1] == self.ncp_t:
                    try:
                        output[(state_name,key)] = xvar[key].value
                    except KeyError:
                        print('something went wrong calling get_state_trajectory')
                        continue
                else:
                    continue
        return output    
        
    def set_reference_control_profile(self,control_profile):
        # input_trajectory = {('state',(fe,j)):value} ---> this format
        self.reference_control_profile = control_profile
        
    def get_control_profile(self,d_mod):
        output = {}
        for control_name in self.u:
            u = getattr(d_mod,control_name)
            for key in u.index_set():
                try:
                    output[(control_name,key)] = u[key].value
                except KeyError:
                    print('something went wrong calling get_state_trajectory')
                    continue
        return output           
     
         
    def load_reference_trajectories(self):
        for x in self.x:
            xvar = getattr(self.olnmpc, x)
            for key in xvar.index_set():
                if key[1] != self.ncp_t:# implicitly assuming RADAU nodes
                    continue 
                if type(key[2:]) == tuple:
                    j = key[2:]
                else:
                    j = (key[2:],)
                fe = key[0] 
                self.olnmpc.xmpc_ref_nmpc[fe,self.xmpc_key[(x,j)]] = self.reference_state_trajectory[x,(fe + self._c_it,self.ncp_t)+j]
                self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0/(max(abs(self.reference_state_trajectory[x,(fe + self._c_it,self.ncp_t)+j]),1e-3))**2

        for u in self.u:    
            uvar = getattr(self.olnmpc, u)
            for j in uvar.index_set():
                fe = j[0]
                try:
                    self.olnmpc.umpc_ref_nmpc[fe,self.umpc_key[u,j[-1]]] = self.reference_control_profile[u,(fe+self._c_it,j[-1])]
                    self.olnmpc.R_nmpc[self.umpc_key[u,j[-1]]] = 1.0/(max(abs(self.reference_control_profile[u,(fe+self._c_it,j[-1])]),1e-3))**2
                except KeyError:
                    self.olnmpc.umpc_ref_nmpc[fe,self.umpc_key[u,j[-1]]] = self.reference_control_profile[u,(self.nfe_t_0,j[-1])]
                    self.olnmpc.R_nmpc[self.umpc_key[u,j[-1]]] = 1.0/(max(abs(self.reference_control_profile[u,(self.nfe_t_0,j[-1])]),1e-3))**2

                    
    def set_regularization_weights(self, K_w = 1.0, Q_w = 1.0, R_w = 1.0):
        if self.obj_type == 'economic':
            for i in self.olnmpc.fe_t:
                self.olnmpc.K_w_nmpc[i] = K_w
        else:
            for i in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[i] = Q_w
                self.olnmpc.R_w_nmpc[i] = R_w
                self.olnmpc.K_w_nmpc[i] = K_w        
            
    """ preparation and initialization """       
    def set_predicted_state(self,m):
        self.predicted_state = {}
        for _var in m.component_objects(Var, active=True):
            for _key in _var.index_set():
                try:
                    self.predicted_state[(_var.name,_key)] = _var[_key].value
                except KeyError:
                    continue

    def cycle_ics(self,nmpc_as=False):
        for x in self.x:
            xic = getattr(self.olnmpc, x+'_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = self.curr_pstate[(x,j)] if nmpc_as else self.curr_rstate[(x,j)]
                
    def cycle_nmpc(self,**kwargs):
        m = self.olnmpc if self._c_it > 1 else self.recipe_optimization_model
        initialguess = kwargs.pop('init',self.store_results(m))
        # cut-off one finite element
        self.nfe_t -= 1
        
        # choose which type of nmpc controller is used
        if self.obj_type == 'economic':
            self.create_enmpc()
        else:
            self.create_nmpc()
            
        # initialize the new problem with shrunk horizon by the old one
        for _var in self.olnmpc.component_objects(Var, active=True):
            for _key in _var.index_set():
                if not(_key == None or type(_key) == str): # if the variable is time invariant scalar skip this 
                    if type(_key) == tuple and type(_key[0]) == int: # for variables that are indexed not only by number of finite element      
                        if _key[0] == self.min_horizon and self.nfe_t == self.min_horizon:
                            shifted_element = (_key[0],)
                        else:
                            shifted_element = (_key[0] + 1,)   # shift finite element by 1
                        aux_key = (_var.name,shifted_element + _key[1:-1] + (1,)) # adjust key
                    elif type(_key) == int: # for variables that are only indexed by number of finite element
                        if _key == self.min_horizon and self.nfe_t == self.min_horizon:
                            shifted_element = _key
                        else:
                            shifted_element = _key + 1      # shift finite element by 1
                            aux_key = (_var.name,shifted_element)
                    else:
                        aux_key = (_var.name,_key)
                else:
                    aux_key = (_var.name,_key)
                try:
                    _var[_key] = initialguess[aux_key] #if initialguess[aux_key] != None else 1.0
                except KeyError:
                    if self.multimodel:
                        try:
                            if type(aux_key[1]) == int:
                                aux_key = (aux_key[0],1)
                            else:
                                aux_key = (aux_key[0],aux_key[1][:-1] + (1,))
                            _var[_key] = initialguess[aux_key]
                        except KeyError:
                            continue
                    else:    
                        continue 
        
        # adapt parameters by on-line estimation
        if self.adapt_params and self._c_it > 1:
            for index in self.curr_epars:
                p = getattr(self.olnmpc, index[0])
                key = index[1] if index[1] != () else None
                p[key].value = self.curr_epars[index]
 
        # set initial value parameters in model olnmpc
        # set according to predicted state by forward simulation
        # a) values will not be overwritten in case advanced step is used
        # b) values will be overwritten by call of add_noise() in case advanced step is not used
        for x in self.x:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = self.curr_pstate[(x,j)]
                    
    """ create models """ 
    def create_enmpc(self):
        self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t, scenario_tree = self.st, s_max = self.s_used, robust_horizon = self.nr)
        self.olnmpc.name = "olnmpc (Open-Loop eNMPC)"
        self.olnmpc.create_bounds() 
        self.create_tf_bounds(self.olnmpc)
        self.olnmpc.clear_aux_bounds()
        
         # Regularization Steps
        self.olnmpc.K_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True)
        expression = 0.0
        if self.delta_u:
            for u in self.u:    
                control = getattr(self.olnmpc, u)
                for key in control.index_set():
                    if key[0] > 1:
                        aux_key = (key[0]-1,key[1])
                        expression += self.olnmpc.K_w_nmpc[key[0]]*(control[aux_key] - control[key])**2.0
                    else:
                        expression += self.olnmpc.K_w_nmpc[1]*(self.nmpc_trajectory[self._c_it,u] - control[1,1])**2.0
              
        # generate the expressions for the objective function
        self.olnmpc.uK_expr_nmpc = Expression(expr = expression)
        self.olnmpc.eobj.expr += 1/self.s_used * self.olnmpc.uK_expr_nmpc
        
    def create_nmpc(self):
        self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t, scenario_tree = self.st, s_max = self.s_used, robust_horizon = self.nr)            
        self.olnmpc.name = "olnmpc (Open-Loop NMPC)"
        self.olnmpc.create_bounds()
        self.create_tf_bounds(self.olnmpc)
        self.olnmpc.clear_aux_bounds()
        if not(hasattr(self.olnmpc, 'ipopt_zL_in')):
            self.olnmpc.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            self.olnmpc.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            self.olnmpc.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            self.olnmpc.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            self.olnmpc.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
       
        # preparation for tracking objective function
        self.xmpc_l = {i:[] for i in self.olnmpc.fe_t} 
             # dictionary that includes a list of the state variables 
             # for each finite element at cp = ncp_t 
             # -> {finite_element:[list of states x(j)[finite_element, ncp]]} 
        
        self.xmpc_key = {} 
             # encodes which state variable takes which index in the list stored in xmpc_l
        # works
        k = 0
        for x in self.x:
            xvar = getattr(self.olnmpc,x)
            for key in xvar.index_set():
                if key[1] == self.ncp_t: # implicitly assumed RADAU nodes
                    self.xmpc_l[key[0]].append(xvar[key])
                else:
                    continue
                if key[0] == 1:
                    self.xmpc_key[(x,key[2:])] = k
                    k += 1
        
        self.umpc_l = {i:[] for i in self.olnmpc.fe_t}
        self.umpc_key = {}
        k = 0
        for u in self.u:
            uvar = getattr(self.olnmpc,u)
            for key in uvar.index_set():
                self.umpc_l[key[0]].append(uvar[key])
                if key[0] == 1: # only relevant for the first run, afterwards repeating progressio
                    self.umpc_key[(u,key[1])] = k
                    k += 1
        
        
        # Parameter Sets that help to index the different states/controls for 
        # tracking terms according to xmpc_l/umpc_l + regularization for control steps delta_u
        self.olnmpc.xmpcS_nmpc = Set(initialize=[i for i in range(0, len(self.xmpc_l[1]))])
        self.olnmpc.umpcS_nmpc = Set(initialize=[i for i in range(0, len(self.umpc_l[1]))])
        # A: The reference trajectory
        self.olnmpc.xmpc_ref_nmpc = Param(self.olnmpc.fe_t, self.olnmpc.xmpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.umpc_ref_nmpc = Param(self.olnmpc.fe_t, self.olnmpc.umpcS_nmpc, initialize=0.0, mutable=True)
        # B: Weights for the different states (for x (Q) and u (R))
        self.olnmpc.Q_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=1.0, mutable=True)  
        self.olnmpc.R_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=1.0, mutable=True)
        # C: Weights for the different finite elements (time dependence) (for x (Q) and u (R))
        self.olnmpc.Q_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True) 
        self.olnmpc.R_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True) 
        self.olnmpc.K_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True)
        
        # regularization on control step
        expression = 0.0
        if self.delta_u:
            for u in self.u:    
                control = getattr(self.olnmpc, u)
                for key in control.index_set():
                    if key[0] > 1:
                        aux_key = (key[0]-1,key[1])
                        expression += self.olnmpc.K_w_nmpc[key[0]]*(control[aux_key] - control[key])**2.0
                    else:# in first stage the controls coincide anyways
                        expression += self.olnmpc.K_w_nmpc[1]*(self.nmpc_trajectory[self._c_it,u] - control[1,1])**2.0
              
        # generate the expressions for the objective function
        self.olnmpc.uK_expr_nmpc = Expression(expr = expression)
        
        # A: sum over state tracking terms
        self.olnmpc.xQ_expr_nmpc = Expression(expr=sum(
        sum(self.olnmpc.Q_w_nmpc[fe] *
            self.olnmpc.Q_nmpc[k] * (self.xmpc_l[fe][k] - self.olnmpc.xmpc_ref_nmpc[fe,k])**2 for k in self.olnmpc.xmpcS_nmpc)
            for fe in range(1, self.nfe_t+1)))

        # B: sum over control tracking terms
        self.olnmpc.xR_expr_nmpc = Expression(expr=sum(
        sum(self.olnmpc.R_w_nmpc[fe] *
                self.olnmpc.R_nmpc[k] * (self.umpc_l[fe][k] - self.olnmpc.umpc_ref_nmpc[fe,k]) ** 2 for k in
                self.olnmpc.umpcS_nmpc) for fe in range(1, self.nfe_t + 1)))
        
        # deactive economic obj function (used in recipe optimization)
        self.olnmpc.eobj.deactivate()
        
        # declare/activate tracking obj function
        self.olnmpc.objfun_nmpc = Objective(expr = (self.olnmpc.eobj.expr +
                                                    + self.olnmpc.xQ_expr_nmpc 
                                                    + self.olnmpc.xR_expr_nmpc 
                                                    + self.olnmpc.uK_expr_nmpc))
        
    def create_noisy_model(self):
        self.noisy_model.tf = self.recipe_optimization_model.tf.value
        self.noisy_model.tf.fixed = True
        self.noisy_model.create_bounds()
        self.noisy_model.eobj.deactivate()
        # improve here and deactivate automatically
        self.noisy_model.deactivate_epc()
        self.noisy_model.deactivate_pc()
        
        # 1. remove initial conditions from the model
        k = 0
        for x in self.x:
            xvar = getattr(self.noisy_model, x)  #: state
            xicc = getattr(self.noisy_model, x + "_icc")
            xicc.deactivate()
            for j in self.x[x]:
                self.xp_l.append(xvar[(1, 0) + j + (1,)])
                self.xp_key[(x, j)] = k
                k += 1
        
        # 2. introduce new variables for noise and reference noise
        self.noisy_model.xS_pnoisy = Set(initialize=[i for i in range(0, len(self.xp_l))])  #: Create set of noisy_states
        self.noisy_model.w_pnoisy = Var(self.noisy_model.xS_pnoisy, initialize=0.0)  #: Model disturbance
        self.noisy_model.w_ref = Param(self.noisy_model.xS_pnoisy,initialize=0.0, mutable=True)
        self.noisy_model.Q_pnoisy = Param(self.noisy_model.xS_pnoisy, initialize=1, mutable=True)
        # 3. redefine Objective: Find the noise that it is close to the randomly generated one but results in consistent initial conditions
        self.noisy_model.obj_fun_noisy = Objective(sense=minimize,
                                  expr=0.5 *
                                      sum(self.noisy_model.Q_pnoisy[k] * (self.noisy_model.w_pnoisy[k]-self.noisy_model.w_ref[k])**2 for k in self.noisy_model.xS_pnoisy))
    
        # 4. define new initial conditions + noise
        self.noisy_model.ics_noisy = ConstraintList()
        for x in self.x:
            xvar = getattr(self.noisy_model, x)  #: state
            xic = getattr(self.noisy_model, x + "_ic")
            for j in self.x[x]:
                xkey = (1,0,1) if j == () else (1,0) + j + (1,)
                xickey = None if j == () else j
                expr = xvar[xkey] == xic[xickey] + self.noisy_model.w_pnoisy[self.xp_key[(x,j)]] 
                self.noisy_model.ics_noisy.add(expr)


    """ optimization/simulation calls """
    def recipe_optimization(self, multimodel=False):
        self.journalizer('O',self._c_it,'off-line recipe optimization','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        self.recipe_optimization_model = self.d_mod(self.nfe_t,self.ncp_t,scenario_tree = self.st, s_max = self.s_used, robust_horizon = self.nr)#self.d_mod(self.nfe_t, self.ncp_t, scenario_tree = self.st)
        self.recipe_optimization_model.initialize_element_by_element()       
        self.recipe_optimization_model.create_output_relations()
        self.recipe_optimization_model.create_bounds()
        self.recipe_optimization_model.clear_aux_bounds()
        if multimodel:
            self.recipe_optimization_model.multimodel()
        self.create_tf_bounds(self.recipe_optimization_model)   
                
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["tol"] = 1e-8
        ip.options["linear_solver"] = "ma57"
        ip.options["max_iter"] = 5000
        
        results = ip.solve(self.recipe_optimization_model, tee=True, symbolic_solver_labels=True, report_timing=True)
        if not(str(results.solver.status) == 'ok' and str(results.solver.termination_condition)) == 'optimal':
            print('Recipe Optimization failed!')
            sys.exit()
            
        self.nmpc_trajectory[1,'tf'] = self.recipe_optimization_model.tf[1,1].value # start at 0.0
        for u in self.u:
            control = getattr(self.recipe_optimization_model,u)
            self.nmpc_trajectory[1,u] = control[1,1].value # control input bewtween time step i-1 and i
            self.curr_u[u] = control[1,1].value #
        
        # directly apply state as predicted state --> forward simulation would reproduce the exact same result since no additional measurement is known
        for x in self.x:
            xvar = getattr(self.recipe_optimization_model,x)
            for j in self.x[x]:
                self.curr_pstate[(x,j)] = xvar[(1,self.ncp_t)+j+(1,)].value
                self.initial_values[(x,j)] = xvar[(1,self.ncp_t)+j+(1,)].value                
        
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    
    def get_tf(self,s):
        t = [] 
        for i in self.recipe_optimization_model.fe_t:
            if i == 1:
                t.append(self.recipe_optimization_model.tf[i,s].value)
            else:
                t.append(t[i-1-1] + self.recipe_optimization_model.tf[i,s].value)
        return t
    
    def solve_olnmpc(self):
        self.journalizer('U',self._c_it,'open-loop optimal control problem','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-5
        ip.options["max_iter"] = 1000
        with open("ipopt.opt", "w") as f:
            f.write("print_info_string yes")
        f.close()

        # enable l1-relaxation but set relaxation parameters to 0
        self.olnmpc.eps.unfix()
        for i in self.olnmpc.eps.index_set():
            self.olnmpc.eps[i].value = 0
            
        self.olnmpc.clear_aux_bounds() #redudant I believe
        
        results = ip.solve(self.olnmpc,tee=True)
        self.olnmpc.solutions.load_from(results)
        #self.olnmpc.write_nl()         
        if not(str(results.solver.status) == 'ok' and str(results.solver.termination_condition) == 'optimal'):
            print('olnmpc failed to converge')
            
        # save relevant results
        # self._c_it + 1 holds always if solve_olnmpc called at the correct time
        self.nmpc_trajectory[self._c_it,'solstat'] = [str(results.solver.status),str(results.solver.termination_condition)]
        self.nmpc_trajectory[self._c_it+1,'tf'] = self.nmpc_trajectory[self._c_it,'tf'] + self.olnmpc.tf[1,1].value
        self.nmpc_trajectory[self._c_it,'eps'] = [self.olnmpc.eps[1,1].value,self.olnmpc.eps[2,1].value,self.olnmpc.eps[3,1].value]  
        if self.obj_type == 'economic':
            self.nmpc_trajectory[self._c_it,'obj_value'] = value(self.olnmpc.eobj)
        else:
            self.nmpc_trajectory[self._c_it,'obj_value'] = value(self.olnmpc.objfun_nmpc)
  
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.nmpc_trajectory[self._c_it+1,u] = control[1,1].value # control input between timestep i-1 and i
        
        # initial state is saved to keep track of how good the estimates are
        # here only self.iteration since its the beginning of the olnmpc run
        for x in self.x:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                self.nmpc_trajectory[self._c_it,(x,j)] = xic[xkey].value   
                    
        # save the control result as current control input
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.curr_u[u] = control[1,1].value
         
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    
    def add_noise(self,var_dict):
        """ Args:
                var_dict = {('xname',(xkey,)):standard_deviation}
        """
        self.journalizer('Pl',self._c_it,'consistent noise generation','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        # set time horizon
        self.noisy_model.tf = self.recipe_optimization_model.tf.value # just a dummy value in order to prohibit tf to go to 0
        # to account for possibly inconsistent initial values
        # solve auxilliary problems with only 1 finite element  
        for x in self.x:
            xic = getattr(self.noisy_model, x + '_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = self.curr_rstate[(x,j)] 
        
        # 4. draw random numbers from normal distribution and compute absolute (!)
        #    noise
        noise_init = {}
        for key in var_dict:
            noise_init[key] = np.random.normal(loc=0.0, scale=var_dict[key])
            xic = getattr(self.noisy_model, key[0] + '_ic')
            xkey = None if key[1] == () else key[1]
            noise_init[key] = noise_init[key]*xic[xkey].value
            
        # 5. define the weighting factor based on the standard deviation
            v_i = self.xp_key[key]
            xic = getattr(self.noisy_model, key[0] + '_ic')
            xkey = None if key[1] == () else key[1]
            self.noisy_model.Q_pnoisy[v_i].value = 1.0/(var_dict[key]*xic[xkey].value + 1e-6)
                
        # 6. define the reference noise values w_ref as randomly generated ones 
        #    and set the initial guess to the same value
            self.noisy_model.w_ref[v_i].value = noise_init[key]
            self.noisy_model.w_pnoisy[v_i].value = noise_init[key]
            if var_dict[key] == 0:
                self.noisy_model.w_pnoisy[v_i].fixed = True
            
        
        # 7. solve the problem
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8

        results = ip.solve(self.noisy_model, tee=True)
        
        self.nmpc_trajectory[self._c_it, 'solstat_noisy'] = [str(results.solver.status),str(results.solver.termination_condition)]
        self.nmpc_trajectory[self._c_it, 'obj_noisy'] = value(self.noisy_model.obj_fun_noisy.expr)
        
        # save the new consistent initial conditions 
        for key in var_dict:
            vni = self.xp_key[key]
            self.curr_rstate[key] = self.noisy_model.w_pnoisy[vni].value + self.curr_rstate[key] # from here on it is a noisy real state
            
        # Set new (noisy) initial conditions in model self.olnmpc
        for x in self.x:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                xic[xkey].value = self.curr_rstate[(x,j)]                   
                
        # monitoring
        for key in var_dict:    
            vni = self.xp_key[key]
            self.nmpc_trajectory[self._c_it,'state_noise',key] = self.noisy_model.w_pnoisy[vni].value
            self.nmpc_trajectory[self._c_it,'consistency_adjustment',key] = self.noisy_model.w_pnoisy[vni].value - self.noisy_model.w_ref[vni].value

        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    
    def plant_simulation(self, **kwargs):
        self.journalizer('Pl',self._c_it,'plant simulation','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        m = self.olnmpc if self._c_it > 1 else self.recipe_optimization_model
        initialguess = kwargs.pop('init',self.store_results(m))
        # simulates the current 
        disturbance_src = kwargs.pop('disturbance_src', 'process_noise')
        # options: process_noise --> gaussian noise added to all states
        #          parameter noise --> noise added to the uncertain model parameters
        #                               or different ones
        #          input noise --> noise added to the inputs/controls
        # not exhaustive list, can be adapted/extended as one wishes
        # combination of all the above with noise in the initial point supported
        initial_disturbance = kwargs.pop('initial_disturbance', {(x,j):0.0 for x in self.x for j in self.x[x]})
        parameter_disturbance = kwargs.pop('parameter_disturbance', {})
        state_disturbance = kwargs.pop('state_disturbance', {})
        input_disturbance = kwargs.pop('input_disturbance',{})
        parameter_scenario = kwargs.pop('scenario',{})   
        
        #  generate the disturbance lording to specified scenario
        state_noise = {}
        input_noise = {}
        if disturbance_src == 'process_noise':
            if state_disturbance != {}: 
                for key in state_disturbance:
                    state_noise[key] = np.random.normal(loc=0.0, scale=state_disturbance[key])
                    # potentially implement truncation at 2 sigma
            else:
                for x in self.x: 
                    for j in self.x[x]:
                        state_noise[(x,j)] = 0.0   
            
            for u in self.u:
                input_noise[u] = 0.0     
                
        elif disturbance_src == 'parameter_noise':
            for p in parameter_disturbance:
                disturbed_parameter = getattr(self.plant_simulation_model, p[0]) 
                pkey= None if p[1] == () else p[1]
                if self._c_it == 1:
                    self.nominal_parameter_values[p] = deepcopy(disturbed_parameter[pkey].value)
                                    
                if (self._c_it-1)%parameter_disturbance[p][1] == 0:
                    sigma = parameter_disturbance[p][0]
                    rand = np.random.normal(loc=0.0, scale=sigma)
                    #truncation at 2 sigma
                    if abs(rand) > 2*sigma:
                        rand = -2.0 * sigma if rand < 0.0 else 2.0 *sigma
                    disturbed_parameter[pkey].value = self.nominal_parameter_values[p] * (1 + rand)                 
            
            for x in self.x:
                for j in self.x[x]:
                    state_noise[(x,j)] = 0.0
                    
            for u in self.u:
                input_noise[u] = 0.0
        elif disturbance_src == 'input_disturbance': # input noise only
            for x in self.x:
                for j in self.x[x]:
                    state_noise[(x,j)] = 0.0
            for u in self.u:
                input_noise[u] = np.random.normal(loc=0.0, scale=input_disturbance[u])
        elif disturbance_src == 'parameter_scenario':
            for p in parameter_scenario:
                disturbed_parameter = getattr(self.plant_simulation_model, p[0])
                pkey = None if p[1] == () else p[1]
                if self._c_it == 1:
                    self.nominal_parameter_values[p] = deepcopy(disturbed_parameter[pkey].value)
                disturbed_parameter[pkey].value = self.nominal_parameter_values[p]*(1 + parameter_scenario[p])
                
            for x in self.x:
                for j in self.x[x]:
                    state_noise[(x,j)] = 0.0
                    
            for u in self.u:
                input_noise[u] = 0.0        
        else:
            print('NO DISTURBANCE SCENARIO SPECIFIED, NO NOISE ADDED ANYWHERE')                     
            for x in self.x:
                for j in self.x[x]:
                        state_noise[(x,j)] = 0.0
            for u in self.u:
                input_noise[u] = 0.0
                    
        
        if self._c_it == 1:
            for x in self.x:
                xic = getattr(self.plant_simulation_model,x+'_ic')
                xvar = getattr(self.plant_simulation_model,x)
                for j in self.x[x]:
                    xkey = None if j == () else j
                    xic[xkey].value = xic[xkey].value * (1.0 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j)]))
                    # initialization
                    for k in range(0,self.ncp_t+1):
                        xvar[(1,k)+j+(1,)].value = xic[xkey].value
            
            # initialization of the simulation (initial guess)               
            for var in self.plant_simulation_model.component_objects(Var, active=True):
                var_ref = getattr(self.recipe_optimization_model, var.name)
                for key in var.index_set():
                    if var[key].fixed:
                        break
                    else:
                        var[key].value = var_ref[key].value
        else:       
            # shifting initial conditions
            # adding state noise if specified    
            for x in self.x:
                xic = getattr(self.plant_simulation_model,x+'_ic')
                x_var = getattr(self.plant_simulation_model,x)
                for j in self.x[x]:
                    xkey = None if j == () else j
                    xic[xkey].value = x_var[(1,self.ncp_t)+j+(1,)].value * (1.0 + state_noise[(x,j)])#+ noise # again tailored to RADAU nodes
                    
            # initialization of simulation
            for var in self.plant_simulation_model.component_objects(Var, active=True):
                try:
                    var_ref = getattr(self.olnmpc, var.name)
                except:
                    # catch that plant simulation includes quantities (Outputs)
                    # that are not relevant for optimal control problem (therefore removed)
                    continue
                for key in var.index_set():
                    if var[key].fixed:
                        break
                    else:
                        var[key].value = var_ref[key].value
            
        # result is the previous olnmpc solution 
        #    --> therefore provides information about the sampling interval
        # 1 element model for forward simulation
        self.plant_simulation_model.tf[1,1].fix(initialguess['tf', (1,1)])

        for u in self.u:
            control = getattr(self.plant_simulation_model,u)
            control[1,1].fix(initialguess[u,(1,1)]*(1.0+input_noise[u]))
        
        self.plant_simulation_model.equalize_u(direction="u_to_r") 

        # probably redundant
        self.plant_simulation_model.clear_all_bounds()      
        
        # solve statement
        ip = SolverFactory("asl:ipopt")
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 1000
        
        out = ip.solve(self.plant_simulation_model, tee=True)

        if [str(out.solver.status), str(out.solver.termination_condition)] != ['ok','optimal']:
            self.plant_simulation_model.create_bounds()
            self.plant_simulation_model.clear_aux_bounds()
            out = ip.solve(self.plant_simulation_model, tee = True)
        
        self.plant_trajectory[self._c_it,'solstat'] = [str(out.solver.status), str(out.solver.termination_condition)]
        self.plant_trajectory[self._c_it,'tf'] = self.plant_simulation_model.tf[1,1].value
    
        # safe results of the plant_trajectory dictionary {number of iteration, (x,j): value}
        for u in self.u:
            control = getattr(self.plant_simulation_model,u)
            self.plant_trajectory[self._c_it,u] = control[1,1].value #(control valid between 0 and tf)
        for x in self.x:
            xvar = getattr(self.plant_simulation_model, x)
            for j in self.x[x]:
                    self.plant_trajectory[self._c_it,(x,j)] = xvar[(1,3)+j+(1,)].value 
                    # setting the current real value w/o measurement noise
                    self.curr_rstate[(x,j)] = xvar[(1,3)+j+(1,)].value 
                    
        # monitoring
        for pc in self.path_constraints:
            pc_var = getattr(self.plant_simulation_model, pc)
            for index in pc_var.index_set():
                self.pc_trajectory[(pc,(self._c_it,index[1:-1]))] = pc_var[index].value

        self.monitor[self._c_it] = {}
        for poi in self.poi:
            poi_var = getattr(self.plant_simulation_model, poi)
            for index in poi_var.index_set():
                self.monitor[self._c_it][(poi,index)] = poi_var[index].value
                
        for cp in range(self.ncp_t+1):
            self.pc_trajectory[('tf',(self._c_it,cp))] = self.plant_simulation_model.tau_i_t[cp]*self.plant_simulation_model.tf[1,1].value
        
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    
    def forward_simulation(self):
        self.journalizer('P',self._c_it,'forward simulation','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        # simulates forward from the current measured or (!) esimated state
        # using the sensitivity based updated control inputs
        
        # IMPORTANT:
        # --> before calling forward_simulation() need to call dot_sens to perform the update
    
        # curr_state_info:
        # comprises both cases where state is estimated and directly measured
        current_state_info = {}
        for key in self.curr_pstate:
            current_state_info[key] = self.curr_pstate[key] - self.curr_state_offset[key]
           
        # save updated controls to current_control_info
        # apply clamping strategy if controls violate physical bounds#
        current_control_info = {}
        for u in self.u:
            control = getattr(self.olnmpc, u)
            current_control_info[u] = control[1,1].value
            
        # change tf via as as well?
        self.forward_simulation_model.tf[1,1].fix(self.olnmpc.tf[1,1].value) 

        if self.adapt_params and self._c_it > 1:
            for index in self.curr_epars:
                p = getattr(self.forward_simulation_model, index[0])
                key = index[1] if index[1] != () else None
                p[key].value = self.curr_epars[index]
                
        # general initialization
        # not super efficient but works
        for var in self.olnmpc.component_objects(Var, active=True):
            xvar = getattr(self.forward_simulation_model, var.name)
            for key in var.index_set():
                try:
                    if key == None or type(key) == int:
                        xvar[key].value = var[key].value
                    elif type(key) == tuple:
                        if len(key) > 2:
                            xvar[key].value = var[(1,self.ncp_t)+key[2:]].value
                        else:
                            xvar[key].value = var[(1,self.ncp_t)].value
                except KeyError:
                    # catches exceptions for trying to access nfe>1 for self.forward_simulation_model
                    continue
        
            
        # 1 element model for forward simulation
        # set initial point (measured or estimated)
        #       --> implicit assumption: RADAU nodes
        for x in self.x:
            xic = getattr(self.forward_simulation_model,x+'_ic')
            xvar = getattr(self.forward_simulation_model,x)
            for j in self.x[x]:
                if j == (): 
                    xic.value = current_state_info[(x,j+(1,))] #+ noise # again tailored to RADAU nodes
                else:
                    xic[j].value = current_state_info[(x,j+(1,))] # + noise # again tailored to RADAU nodes
                # for initialization: take constant values + leave time invariant values as is!
                for k in range(0,self.ncp_t+1):# use only 
                    xvar[(1,k)+j+(1,)].value = current_state_info[(x,j+(1,))]
                    

        # set and fix control as provided by olnmpc/advanced step update
        for u in self.u:
            control = getattr(self.forward_simulation_model,u)
            control[1,1].fix(current_control_info[u])
            self.forward_simulation_model.equalize_u(direction="u_to_r") # xxx
        
        self.forward_simulation_model.clear_all_bounds()
        
        # solve statement
        ip = SolverFactory("asl:ipopt")
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 1000

        out = ip.solve(self.forward_simulation_model, tee=True)
        self.simulation_trajectory[self._c_it,'obj_fun'] = 0.0
        
        if [str(out.solver.status), str(out.solver.termination_condition)] != ['ok','optimal']:
            self.forward_simulation_model.clear_all_bounds()
            out = ip.solve(self.forward_simulation_model, tee = True, symbolic_solver_labels=True)

        self.simulation_trajectory[self._c_it,'solstat'] = [str(out.solver.status), str(out.solver.termination_condition)]
        
        # save the simulated state as current predicted state
        # implicit assumption of RADAU nodes
        for x in self.x:
            xvar = getattr(self.forward_simulation_model, x)
            for j in self.x[x]:
                    self.curr_pstate[(x,j)] = xvar[(1,self.ncp_t)+j+(1,)].value  # implicit assumption of RADAU nodes
  
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    

    """ sensitivity-based updates """
    def create_suffixes_nmpc(self, first_iter = False):
        self.journalizer('P',self._c_it,'creating k_aug suffixes','')
        """Creates the required suffixes for the olnmpc problem"""
        if first_iter:
            m = self.recipe_optimization_model
        else:
            m = self.olnmpc
        if hasattr(m, "npdp"):
            pass
        else:
            m.npdp = Suffix(direction=Suffix.EXPORT)
        if hasattr(m, "dof_v"):
            pass
        else:
            m.dof_v = Suffix(direction=Suffix.EXPORT)

        for u in self.u:
            uv = getattr(m, u)
            uv[1,1].set_suffix_value(m.dof_v, 1)
    
    def compute_offset_state(self, src_kind="estimated"):
        """Missing noisy"""
        self.journalizer('U',self._c_it,'state offset computation','')
        if src_kind == "estimated":
            for x in self.x:
                for j in self.x[x]:
                    self.curr_state_offset[(x, j)] = self.curr_pstate[(x, j)] - self.curr_estate[(x, j)]
        elif src_kind == "real":
            for x in self.x:
                for j in self.x[x]:
                    self.curr_state_offset[(x, j)] = self.curr_pstate[(x, j)] - self.curr_rstate[(x, j)]
                    
        self.nmpc_trajectory[self._c_it,'state_offset'] = self.curr_state_offset

    def sens_k_aug_nmpc(self):
        self.journalizer("PR", self._c_it, "NLP sensitivity evaluation", self.olnmpc.name + "using k_aug sensitivity")
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        self.olnmpc.ipopt_zL_in.update(self.olnmpc.ipopt_zL_out)
        self.olnmpc.ipopt_zU_in.update(self.olnmpc.ipopt_zU_out)
        
        if hasattr(self.olnmpc, "f_timestamp"):
            self.olnmpc.f_timestamp.clear()
        else:
            self.olnmpc.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                             datatype=Suffix.INT)

        self.olnmpc.set_suffix_value(self.olnmpc.f_timestamp, self.int_file_nmpc_suf)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)
        # additional helpful k_aug options
        #self.k_aug_sens.options["no_inertia"] = ""
        #self.k_aug_sens.options["no_barrier"] = ""
        #self.k_aug_sens.options['target_log10mu'] = -5.7
        self.k_aug_sens.options['no_scale'] = ""
        
        results = self.k_aug_sens.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)
        ftimings = open("timings_k_aug.txt", "r")
        s = ftimings.readline()
        ftimings.close()
        self._k_timing = s.split() 
        
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)

    def sens_dot_nmpc(self):
        self.journalizer("U", self._c_it, "sensitivity-based update", "sens_dot_nmpc")
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        if hasattr(self.olnmpc, "npdp"):
            self.olnmpc.npdp.clear()
        else:
            self.olnmpc.npdp = Suffix(direction=Suffix.EXPORT)

        for x in self.x:
            con = getattr(self.olnmpc, x + "_icc")
            for j in self.x[x]:
                xkey = None if j == () else j
                con[xkey].set_suffix_value(self.olnmpc.npdp, self.curr_state_offset[(x, j)])

        if hasattr(self.olnmpc, "f_timestamp"):
            self.olnmpc.f_timestamp.clear()
        else:
            self.olnmpc.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        self.olnmpc.set_suffix_value(self.olnmpc.f_timestamp, self.int_file_nmpc_suf)

        self.olnmpc.f_timestamp.display(ostream=sys.stderr)

        # SAVE OLD
        #before_dict = {}
        for u in self.u:
            control = getattr(self.olnmpc,u)
            #before_dict[u] = control[1,1].value
            
        results = self.dot_driver.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)

        # SAVE NEW
        #after_dict = {}
        #difference_dict = {}
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            #after_dict[u] = control[1,1].value
            #difference_dict[u] = after_dict[u] - before_dict[u]


        ftiming = open("timings_dot_driver.txt", "r")
        s = ftiming.readline()
        ftiming.close()
        k = s.split()
        self._dot_timing = k[0]
        
        # augmented by flho
           
        # save updated controls to current_control_info
        # apply clamping strategy if controls violate physical bounds
        for u in self.u:
            control = getattr(self.olnmpc, u)
            if control[1,1].value > control[1,1].ub:
                if control[1,1].ub == None: # number > None == True 
                    pass
                else:
                    control[1,1] = control[1,1].ub
            elif control[1,1].value < control[1,1].lb: #  number < None == False
                control[1,1].value = control[1,1].lb
            else:
                pass
        
        #if abs(self.olnmpc.u2[1,1].value - self.nmpc_trajectory[self._c_it+1,'u2']) > 1.0:
        #    print('solution way off')
        #    print(self.curr_state_offset)
        #    sys.exit()
            
        #applied = {}
        for u in self.u:
            control = getattr(self.olnmpc, u)
            self.curr_u[u] = control[1,1].value    
            #applied[u] = control[1,1].value
            
        #return before_dict, after_dict, difference_dict, applied
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    
              


    """ on-line scenario-tree adaption """
    def scenario_tree_generation(self, pc = [], epc = [], **kwargs):
        """ procedure has to be called before self.cycle_nmpc! """
        # sensitivity-based worst-case scenario detection
        # either hyperrectangle or hyperellipsoid
        # for hyperrectangles:
        # solve for all constraints (all timepoints) the optimization problem:
        #           max ds/dp^T * delta_p
        #    s.t. 
        #           delta_p_min <= delta_p <= delta_p_max
        #           
        # solution analytically derivable:
        #       delta_p^*_i =  delta_p_max if ds/dp_i > 0 else delta_p_min
    
        # Iteratively populate scenarios:
            # include scenario that would result in maximum constraint violation in first order approximation:
            #           - approx. violation: -s + \\ scaling ds/dp \\_{1}
            #           - always a vertex
            #
            # exclude that vertex for future scenarios
            #           
            # repeat until as many scenarios as possible are included
        
        self.journalizer('P',self._c_it,'scenario-tree generation','')
        start = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        # inputs
        bounds = kwargs.pop('par_bounds',{}) # uncertainty bounds
        crit = kwargs.pop('crit','con')
        noisy_ics = kwargs.pop('noisy_ics',{})
        uncertain_params = kwargs.pop('uncertain_params',self.p_noisy)
        m = self.olnmpc if self._c_it > 1 else self.recipe_optimization_model
        m = kwargs.pop('m',m)        
        
        # prepare sensitivity computation
        m.eps_pc.fix()
        m.eps.fix()
        for u in self.u:
            u_var = getattr(m,u)
            u_var.fix()
        m.tf.fix()
        m.clear_all_bounds()
        
        for var in m.ipopt_zL_in:
            var.set_suffix_value(m.ipopt_zL_in, 0.0)
                
        for var in m.ipopt_zU_in:
            var.set_suffix_value(m.ipopt_zU_in, 0.0)
        
        # deactivate nonanticipativity
        for u in self.u:
            non_anti = getattr(m, 'non_anticipativity_' + u)
            non_anti.deactivate()
        # deactivate fixed element size
        m.fix_element_size.deactivate()
        # deavtivate non_anticipativity for tf
        m.non_anticipativity_tf.deactivate()
        try:
            m.con1.deactivate()
            m.con2.deactivate()
            m.con3.deactivate()
            m.con4.deactivate()
        except:
            pass
        #
        try:
            m.epi.deactivate()
        except:
            pass
            
        # set suffixes
        m.var_order = Suffix(direction=Suffix.EXPORT)
        m.dcdp = Suffix(direction=Suffix.EXPORT)
        
        # cols = variables for which senstivities shall be computed
        i = 0
        cols ={}
        for p in uncertain_params:
            for key in uncertain_params[p]:
                dummy = 'dummy_constraint_p_' + p + '_' + str(key[0]) if key != () else 'dummy_constraint_p_' + p
                dummy_con = getattr(m, dummy)
                for index in dummy_con.index_set():
                    #index[0] = time_step \in {2, ... ,nr+1}
                    #index[-1] = scenario \in {1, ... , s_per_branch}
                    #only for nominal scenario otw. very intricat to implement
                    if index[0] > 1 and \
                       index[0] < self.nr + 2 and \
                       index[-1] == 1:
                        m.dcdp.set_value(dummy_con[index], i+1)
                        cols[i] = (p,key+index)
                        i += 1
         
        for p in noisy_ics:
            for key in noisy_ics[p]:
                dummy = 'dummy_constraint_p_' + p + '_' + str(key[0]) if key != () else 'dummy_constraint_p_' + p
                dummy_con = getattr(m, dummy)
                for index in dummy_con.index_set():
                    if index[-1] == 1 and ((index[0] == 2) or index[0] == self.nfe_t):
                        #index[0] = time_step --> 2
                        #index[-1] = scenrio (only nominal scenario)
                        m.dcdp.set_value(dummy_con[index], i+1)
                        cols[i] = (p,key+index)
                        i += 1
                                                 
        # column i in sensitivity matrix corresponds to paramter p
        cols_r = {value:key for key, value in cols.items()}
        tot_cols = i
        
        i = 0
        rows = {}
        #sensitivities of path constraints
        for c in pc:
            s = getattr(m, 's_'+c)
            for index in s.index_set():
                if not(s[index].stale): # only take
                    #index[0] = time_step \in {2, ... ,nr+1}
                    #index[1] = collocation point: all of them here  (alternatively ncp_t: consider only endpoint of interval)
                    #index[-1] = scenario: only nominal scenario as base for linearization
                    #index[2:-1] = additional indices: all there are
                    if index[0] > 1 and \
                       index[0] < self.nr + 1 and \
                       index[-1] == 1: #
                        m.var_order.set_value(s[index], i+1)
                        rows[i] = ('s_'+ c,index)
                        i += 1                
        # endpoint constraints only in last iteration
        # can consider epc in every stage
        #if self._c_it == self.nfe_t_0 - 1: 
        for c in epc:
            s = getattr(m, 's_'+c)
            for index in s.index_set():
                if not(s[index].stale): # only take
                    nominal = True if (type(index) == tuple and index[-1] == 1) or index==1 else False
                    if nominal:
                        # only nominal scenario as base for linearization
                        m.var_order.set_value(s[index], i+1)
                        index = index if type(index)==tuple else (index,)
                        rows[i] = ('s_'+ c,index)
                        i += 1
                        
        # row j in sensitivity matrix corresponds to rhs of constraint x 
        rows_r = {value:key for key, value in rows.items()}
        tot_rows = i

        # compute sensitivity matrix (ds/dp , rows = s const., cols = p const.)
        k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
        k_aug.options["compute_dsdp"] = ""
        k_aug.options["no_scale"] = ""
        k_aug.solve(m, tee=True)
            
        m.eps.unfix()
        m.eps_pc.unfix()
        for u in self.u:
            u_var = getattr(m,u)
            u_var.unfix()
        m.tf.unfix()
        m.create_bounds()
        self.create_tf_bounds(m)
        m.clear_aux_bounds()
        
        # activate non_anticipativity
        for u in self.u:
            non_anti = getattr(m, 'non_anticipativity_' + u)
            non_anti.activate()
        # activate non_anticipativity for tf
        m.non_anticipativity_tf.activate()     
        # activate element size
        m.fix_element_size.activate()    
        
        # read sensitivity matrix from file "dxdp_.dat"
        # rows correspond to variables/constraints
        # cols correspond to parameters
        sens = np.zeros((tot_rows,tot_cols))
        with open('dxdp_.dat') as f:
            reader = csv.reader(f, delimiter="\t")
            i = 1
            for row in reader:
                k = 1
                for col in row[1:]:
                    #indices start at 0
                    #different sign since g(x) == -slack
                    # should be -float , bug in k_aug
                    sens[i-1][k-1] = -float(col) 
                    k += 1                       
                i += 1
        
        # compute worst-case scenarios 
        delta_p_wc = {}
        con_vio = {}
        for i in rows: # constraints
            con = rows[i] # ('s_name', index)
            c_name = con[0][2:]
            c_stage = min(self.nr+1,self.nfe_t) if c_name in epc else con[1][0]  
            c_scen = con[1][-1]
            dsdp = sens[i][:].transpose() # gets row i in dsdp, transpose not required but makes clear what is done
            delta_p_wc_iter = {}
            vertex = {}
            aux = 0.0
            for j in cols: # parameters
                par = cols[j] # cols[j] = ('pname', index)
                p = par[0] 
                key = par[1][:-2]
                p_stage = par[1][-2]
                p_scen = par[1][-1]
                # only compute for relevant parameters 
                # distinguish between endpoint and path constraints
                # endpoint constraints are only considered on last stage
                if  ([p_stage,p_scen] == [c_stage,c_scen] and c_name in pc) \
                    or (([p_stage,p_scen] == [self.nr+1,c_scen] or [p_stage,p_scen] == [self.nfe_t,c_scen]) and c_name in epc):
                    p_var = getattr(m, 'p_' + p)
                    # just a preparation for linearizing around different scenarios
                    # will be zero here
                    delta = p_var[key+(p_stage,p_scen)].value - 1.0 
                    # bounds[par][0]: lower bound on delta_p
                    # bounds[par][1]: upper bound on delta_p
                    # shift depending around which scenario is linearized
                    delta_p_wc_iter[p,key] = bounds[(p,key)][0] - delta if dsdp[j] < 0.0 else bounds[(p,key)][1] - delta
                    vertex[p,key] = 'L' if dsdp[j] < 0.0 else 'U'
                    aux += dsdp[j]*delta_p_wc_iter[p,key]
                else:
                    continue
            s = getattr(m, con[0])
            if crit == 'overall':
                con_index = (min(self.nr+1,self.nfe_t),) + con if c_name in epc else con
                con_vio[con_index] = -s[con[1]].value + aux
                delta_p_wc[con_index] = deepcopy(vertex)
            elif crit == 'con':
                if key in con_vio:
                    if con_vio[c_name,c_stage] < -s[con[1]].value + aux:
                        delta_p_wc[c_name,c_stage] = deepcopy(vertex)
                        con_vio[c_name,c_stage] = -s[con[1]].value + aux
                else:
                    delta_p_wc[c_name,c_stage] = deepcopy(vertex)
                    con_vio[c_name,c_stage] = -s[con[1]].value + aux
            else:
                sys.exit('Error: Wrong specification of worst case criterion')
                
        scenarios = {}
        s_branch = {}
        s_stage = {0:1}
        # overall wc 
        #print(delta_p_wc)
        #if crit == 'overall':
        # wc among all constraints
        # i.e, if for the same constraint two scenarios result in higher first-order
        # violations than any scenario for any other constraint both are included
        print(con_vio)
        print('')
        print(delta_p_wc)
        print('')
        for i in range(2,min(self.nr+2,self.nfe_t+1)):
            if crit == 'overall':
                con_vio_copy = {con:con_vio[con] for con in con_vio if con[1][0] == i}
            elif crit == 'con':
                con_vio_copy = {con:con_vio[con] for con in con_vio if con[1] == i}
            else:
                sys.exit('Error: Wrong specification of scenario-tree generation criterion')       
            
            if i != self.nfe_t:
                s_branch_max = int(np.round(self.s_max**(1.0/self.nr)))+1 #if i != self.nfe_t else int(self.s_max/((np.round(self.s_max**((self.nfe_t-1.0)/self.nr)))))+1
            else:
                s_branch_max = int(self.s_max/s_stage[i-2]) + 1 # int always floors
            
            for s in range(2,s_branch_max): # scenario 1 is reserved for the nomnal sscenario
                wc_scenario = max(con_vio_copy,key=con_vio_copy.get)
                print(wc_scenario )
                scenarios[i-1,s] = delta_p_wc[wc_scenario]
                # remove scenario from scenarios:
                con_vio_copy = {key: value for key, value in con_vio_copy.items() if (delta_p_wc[key] != delta_p_wc[wc_scenario])}
                if len(con_vio_copy) == 0:
                    break
            s_branch[i-1] = s
            s_stage[i-1] = s*s_stage[i-2]
                
        self.s_used = s_stage[min(self.nr,self.nfe_t-1)]# nfe_t-1 since it is called before cycle_nmpc
        self.nmpc_trajectory[self._c_it, 's_max'] = self.s_used
    
        print(scenarios)
        print(s_branch)
        # update scenario tree
        self.st = {}
        for i in range(1,self.nfe_t-1+1):#+1
            if i < self.nr + 1:
                for s in range(1,s_stage[i]+1):
                    if s%s_branch[i]==1:
                        self.st[(i,s)] = (i-1,int(np.ceil(s/float(s_branch[i]))),True,{(p,key) : 1.0 for p in self.p_noisy for key in self.p_noisy[p]})
                    else:
                        scenario_key = s%s_branch[i] if s%s_branch[i] != 0 else s_branch[i]
                        if i == 1:# or i == min(self.nr,self.nfe_t):
                            self.st[(i,s)] = (i-1,int(np.ceil(s/float(s_branch[i]))),False,{index: 1.0 + bounds[index][0] if scenarios[i,scenario_key][index]=='L' else 1 + bounds[index][1] \
                                               for index in bounds})    
                        else:
                            self.st[(i,s)] = (i-1,int(np.ceil(s/float(s_branch[i]))),False,{(p,key): 1.0 + bounds[(p,key)][0] if scenarios[i,scenario_key][(p,key)]=='L' else 1 + bounds[(p,key)][1] \
                                               for p in self.p_noisy for key in self.p_noisy[p]})
            else:
                for s in range(1,self.s_used+1):
                    self.st[(i,s)] = (i-1,s,True,self.st[(i-1,s)][3])
        # last stage of new tree includes worst case errors in last decision
#        i = self.nfe_t - 1
#        step = min(self.nfe_t-1,self.nr)
#        for s in range(1,self.s_used+1):
#            if s%s_branch[step]==1:
#                parent_node = s if self.nfe_t - 1 > self.nr else int(np.ceil(s/float(s_branch[step])))
#                self.st[(i,s)] = (i-1,parent_node,True,{(p,key) : 1.0 for p in self.p_noisy for key in self.p_noisy[p]})
#            else:            
#                anchor = True if self.nfe_t - 1 > self.nr else False
#                parent_node = s if self.nfe_t - 1 > self.nr else int(np.ceil(s/float(s_branch[step])))
#                scenario_key = s%s_branch[step] if s%s_branch[step] != 0 else s_branch[step]
#                self.st[(i,s)] = (i-1,parent_node,anchor,{index: 1.0 + bounds[index][0] if scenarios[step,scenario_key][index]=='L' else 1 + bounds[index][1] \
#                                           for index in bounds})
        # save scenario_tree that is used at timestep k = self._c_it + 1 (since st is generated one step in advance)
        self.nmpc_trajectory[self._c_it+1,'st'] = deepcopy(self.st)
        
        end = (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN))
        return (start,end)
    

    """ open-loop considerations """
    def open_loop_simulation(self, sample_size = 10, **kwargs):
        # simulates the current
        # options: process_noise --> gaussian noise added to all states
        #          parameter noise --> noise added to the uncertain model parameters
        #                               or different ones
        #          input noise --> noise added to the inputs/controls
        # not exhaustive list, can be adapted/extended as one wishes
        # combination of all the above with noise in the initial point supporte
        initial_disturbance = kwargs.pop('initial_disturbance', {(x,j):0.0 for x in self.x for j in self.x[x]})
        parameter_disturbance = kwargs.pop('parameter_disturbance', {})
        input_disturbance = kwargs.pop('input_disturbance',{})
        parameter_scenario = kwargs.pop('parameter_scenario',{})
        
        # deactivate constraints
        self.simulation_model = self.d_mod(self.nfe_t, self.ncp_t)
        self.simulation_model.deactivate_epc()
        self.simulation_model.deactivate_pc()
        self.simulation_model.eobj.deactivate()
        self.simulation_model.del_pc_bounds()
        self.simulation_model.clear_all_bounds()
        self.simulation_model.fix_element_size.deactivate()
        
        # load initial guess from recipe optimization
        for var in self.simulation_model.component_objects(Var):
            var_ref = getattr(self.recipe_optimization_model, var.name)
            for key in var.index_set():
                var[key].value = var_ref[key].value
        
        nominal_initial_point = {}
        for x in self.x: 
            xic = getattr(self.recipe_optimization_model, x+'_ic')
            for j in self.x[x]:
                xkey = None if j == () else j
                nominal_initial_point[(x,j)] = xic[xkey].value
                    
        if parameter_disturbance != {} or parameter_scenario != {}:
            par_dict = parameter_disturbance if parameter_disturbance != {} else parameter_scenario[0]
            for p in par_dict:
                key = p[1] if p[1] != () else None
                disturbed_parameter = getattr(self.recipe_optimization_model, p[0])
                self.nominal_parameter_values[p] = disturbed_parameter[key].value
                
        for u in self.u:
            control = getattr(self.simulation_model, u)
            control.fix()
            
        endpoint_constraints = {}
        pc_trajectory = {}
        # set and fix controls + add noise
        for k in range(sample_size):
            self.journalizer('O', k, 'off-line optimal control','')
            pc_trajectory[k] = {}
            if initial_disturbance != {}:
                for x in self.x:
                    xic = getattr(self.simulation_model,x+'_ic')
                    for j in self.x[x]:
                        xkey = None if j == () else j
                        xic[xkey].value = nominal_initial_point[(x,j)] * (1 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j)]))
            if input_disturbance != {}:
                for u in input_disturbance:
                    control = getattr(self.simulation_model, u)
                    for i in range(1,self.nfe_t+1):
                        disturbance_noise = np.random.normal(loc=0.0, scale=input_disturbance[u])
                        control[i,1].value = self.reference_control_profile[u,(i,1)]*(1+disturbance_noise)
            if parameter_disturbance != {}:
                for p in parameter_disturbance:
                    key = p[1] if p[1] != () else None
                    disturbed_parameter = getattr(self.simulation_model, p[0])
                    sigma = parameter_disturbance[p][0]
                    rand = np.random.normal(loc=0.0, scale=sigma)
                    if abs(rand) < 2*sigma:
                        disturbed_parameter[key].value = self.nominal_parameter_values[p] * (1 + rand) 
                    else:
                        if rand < 0:
                           disturbed_parameter[key].value = self.nominal_parameter_values[p] * (1 - 2*sigma)
                        else:
                            disturbed_parameter[key].value = self.nominal_parameter_values[p] * (1 + 2*sigma)
            if parameter_scenario != {}:
                for p in parameter_scenario[k]:
                    key = p[1] if p[1] != () else None
                    disturbed_parameter = getattr(self.simulation_model, p[0])
                    disturbed_parameter[key].value = (1 + parameter_scenario[k][p])*self.nominal_parameter_values[p] 
                        
            
            self.simulation_model.tf.fix()
            self.simulation_model.equalize_u(direction="u_to_r")
            # run the simulation
            ip = SolverFactory("asl:ipopt")
            ip.options["halt_on_ampl_error"] = "yes"
            ip.options["print_user_options"] = "yes"
            ip.options["linear_solver"] = "ma57"
            ip.options["tol"] = 1e-8
            ip.options["max_iter"] = 1000
                
            out = ip.solve(self.simulation_model, tee=True, symbolic_solver_labels=True)
            if  [str(out.solver.status), str(out.solver.termination_condition)] == ['ok','optimal']:
                converged = True
            else:
                converged = False
        
            if converged:
                endpoint_constraints[k] = self.simulation_model.check_feasibility(display=True)
                for pc_name in self.path_constraints:
                    pc_var = getattr(self.simulation_model, pc_name)
                    for fe in self.simulation_model.fe_t:
                        for cp in self.simulation_model.cp:
                            pc_trajectory[k][(pc_name,(fe,(cp,)))] = pc_var[fe,cp,1].value
                            pc_trajectory[k][('tf',(fe,cp))] = self.simulation_model.tau_i_t[cp]*self.recipe_optimization_model.tf[1,1].value
            else:
                sys.exit('Error: Simulation not converged!')
                endpoint_constraints[k] = 'error'
        
        return endpoint_constraints, pc_trajectory
        
    def run(self,regularization_weights={},disturbance_src={},stgen_args={},\
            advanced_step=False,stgen=False):
        ru = {}
        # off-line open-loop control
        ru['recipe_optimization']=self.recipe_optimization()
        self.set_reference_state_trajectory(self.get_state_trajectory(self.recipe_optimization_model))
        self.set_reference_control_profile(self.get_control_profile(self.recipe_optimization_model))
        
        self.create_nmpc() if self.obj_type == 'tracking' else self.create_enmpc() 
        self.load_reference_trajectories() if self.obj_type == 'tracking' else None
    
        # on-line control
        if advanced_step:
            for i in range(1,self.nfe_t_0):
                ru['plant_simulation',i]=self.plant_simulation(**disturbance_src)
                # preparation phase
                if stgen:
                    ru['stgen',i]=self.scenario_tree_generation(**stgen_args)
                self.cycle_nmpc() 
                self.cycle_ics()
                self.load_reference_trajectories() if self.obj_type == 'tracking' else None # loads the reference trajectory in olnmpc problem (for regularization)
                self.set_regularization_weights(**regularization_weights)
                ru['olnmpc',i]=self.solve_olnmpc() # solves the olnmpc problem
                self.create_suffixes_nmpc()
                ru['sens',i]=self.sens_k_aug_nmpc()
                #updating phase
                self.compute_offset_state(src_kind="real")
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
                if stgen:
                    ru['stgen',i]=self.scenario_tree_generation(**stgen_args)
                self.cycle_nmpc()
                self.cycle_ics()
                # updating phase
                self.load_reference_trajectories() if self.obj_type == 'tracking' else None # loads the reference trajectory in olnmpc problem (for regularization)
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
            
#    def reinit(self, initialguess):
#    # initialize new problem with the shifted old one
#        for _var in self.olnmpc.component_objects(Var, active=True):
#            for _key in _var.index_set():
#                if not(_key == None): # if the variable is time invariant scalar skip this 
#                    if type(_key) == tuple: # for variables that are indexed not only by number of finite element               
#                        shifted_element = (_key[0] + 1,)   # shift finite element by 1
#                        aux_key = (_var.name,shifted_element + _key[1:]) # adjust key
#                    else: # for variables that are only indexed by number of finite element
#                        shifted_element = _key + 1      # shift finite element by 1
#                        aux_key = (_var.name,shifted_element)
#                else:
#                    aux_key = (_var.name,_key)
#                try:
#                    _var[_key].value = initialguess[aux_key]
#                except KeyError: # last element will be doubled since exception is thrown if _key[0] == self.nfe_t
#                    continue  
#                
#        for x in self.x:
#            for j in self.x[x]:
#                self.initial_values[(x,j)] = initialguess[(x,(2,0)+j+(1,))] 
#
#        # set initial value parameters in mode olnmpc
#        for x in self.x:
#            xic = getattr(self.olnmpc, x + '_ic')
#            for j in self.x[x]:
#                xkey = None if j == () else j
#                xic[xkey].value = self.initial_values[(x,j)]        

                
        
#    def generate_state_index_dictionary(self):
#        # generates a dictionary = {'name':[indices except fe,cp] if only 1 add. index (j,) if none ()}
#        for x in self.x:
#            self.state_vars[x] = []
#            try:
#                xv = getattr(self.forward_simulation_model, x)
#            except AttributeError:  # delete this
#                continue
#            for j in xv.keys():
#                if j[1] == 0:
#                    if type(j[2:]) == tuple:
#                        self.state_vars[x].append(j[2:])
#                    else:
#                        self.state_vars[x].append((j[2:],))
#                else:
#                    continue        



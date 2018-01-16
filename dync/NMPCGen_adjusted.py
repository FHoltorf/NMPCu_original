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
from main.dync.DynGen_adjusted import DynGen
import numpy as np
import sys, os, time, csv
from six import iterkeys
from copy import deepcopy
__author__ = "David M Thierry @dthierry"

"""Not quite."""


class NmpcGen(DynGen):
    def __init__(self, **kwargs):
        self.stop = False
        DynGen.__init__(self, **kwargs)
        self.int_file_nmpc_suf = int(time.time())+1

        self.ref_state = kwargs.pop("ref_state", None)
        self.u_bounds = kwargs.pop("u_bounds", None)

        # We need a list of tuples that contain the bounds of u
        self.olnmpc = object()

        self.curr_soi = {}  #: Values that we would like to keep track
        self.curr_sp = {}  #: Values that we would like to keep track (from ss2)
        self.curr_off_soi = {}
        self.curr_ur = dict.fromkeys(self.u, 0.0)  #: Controls that we would like to keep track of(from ss2)
#        for k in self.ref_state.keys():
#            self.curr_soi[k] = 0.0
#            self.curr_sp[k] = 0.0

        self.soi_dict = {}  #: State-of-interest
        self.sp_dict = {}  #: Set-point
        self.u_dict = dict.fromkeys(self.u, [])

        # self.res_file_name = "res_nmpc_" + str(int(time.time())) + ".txt"
###########################################################################        
###########################################################################        
###########################################################################        
###########################################################################        
########################################################################### 
        # monitoring
        self.path_constraints = kwargs.pop('path_constraints',[])
        
        self.obj_type = kwargs.pop('obj_type','tracking')
        self.min_horizon = kwargs.pop('min_horizon',0)
        self.n_s = kwargs.pop('n_s',1) # number of scenarios

        self.noisy_model = self.d_mod(1,self.ncp_t)
        self.recipe_optimization_model = object() 
        # model to compute the reference trajectory (open loop optimal control problem with economic obj)
        
        self.reference_state_trajectory = {} # {('state',(fe,j)):value)}
        self.reference_control_trajectory = {} # {('control',(fe,j)):value)}
        self.storage_model = object()
        self.initial_values = {}
        self.nominal_parameter_values = {}

        
        # plant simulation model in order to distinguish between noise and disturbances
        self.plant_simulation_model = self.d_mod(1, self.ncp_t, _t=self._t)
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
        self.forward_simulation_model = self.d_mod(1, self.ncp_t, _t=self._t) # only need 1 element for forward simulation
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
      
        self.iterations = 1
        self.nfe_mhe = 1
        self.nfe_t_0 = 0 # gets set during the recipe optimization

        # parameters covariance matrix
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
        self.weighting_matrix = {}
        
    def open_loop_simulation(self, sample_size = 10, disturbances = {}, **kwargs):
        initial_disturbance = kwargs.pop('initial_disturbance', {(x,j):0.0 for x in self.states for j in self.state_vars})
        parameter_disturbance = kwargs.pop('parameter_disturbance', {})
        
        # deactivate constraints
        self.recipe_optimization_model.deactivate_epc()
        self.recipe_optimization_model.deactivate_pc()
        self.recipe_optimization_model.eobj.deactivate()
        self.recipe_optimization_model.del_pc_bounds()
        self.recipe_optimization_model.clear_aux_bounds()
        
        nominal_initial_point = {}
        for x in self.states: 
            xic = getattr(self.recipe_optimization_model, x+'_ic')
            for j in self.state_vars[x]:
                if j == ():
                    nominal_initial_point[(x,j)] = xic.value
                else:
                    nominal_initial_point[(x,j)] = xic[j].value
                    
        if self.noisy_params:
            for p in parameter_disturbance:
                disturbed_parameter = getattr(self.recipe_optimization_model, p[0])
                self.nominal_parameter_values[p] = disturbed_parameter[p[1]].value
                    
        endpoint_constraints = {}
        pc_trajectory = {}
        # set and fix controls + add noise
        for k in range(sample_size):
            print('#'*20)
            print(' '*5 + 'iter: ' + str(k))
            print('#'*20)
            pc_trajectory[k] = {}
            for u in self.u:
                control = getattr(self.recipe_optimization_model, u)
                control.fix()
                for i in range(1,self.nfe_t+1):
                    disturbance_noise = np.random.normal(loc=0.0, scale=disturbances[u])
                    control[i].value = self.reference_control_trajectory[u,i]*(1+disturbance_noise)
            # perturb initial point
            for x in self.states:
                xic = getattr(self.recipe_optimization_model,x+'_ic')
                for j in self.state_vars[x]:
                    if j == ():
                        xic.value = nominal_initial_point[(x,j)] * (1 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j)]))
                    else:
                        xic[j].value = nominal_initial_point[(x,j)] * (1 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j)]))
            if self.noisy_params:
                for p in parameter_disturbance:
                    disturbed_parameter = getattr(self.recipe_optimization_model, p[0])
                    disturbed_parameter[p[1]].value = self.nominal_parameter_values[p] * (1 + np.random.normal(loc=0.0, scale=parameter_disturbance[p][0]))    
   
            self.recipe_optimization_model.tf.fix()
            self.recipe_optimization_model.equalize_u(direction="u_to_r")
            # run the simulation
            ip = SolverFactory("asl:ipopt")
            ip.options["halt_on_ampl_error"] = "yes"
            ip.options["print_user_options"] = "yes"
            ip.options["linear_solver"] = "ma57"
            ip.options["tol"] = 1e-8
            ip.options["max_iter"] = 3000
                
            out = ip.solve(self.recipe_optimization_model, tee=True)
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
                self.recipe_optimization_model.troubleshooting()
                sys.exit()
                endpoint_constraints[k] = 'error'
        
        return endpoint_constraints, pc_trajectory
    
    def plant_simulation(self,result,first_call = False, **kwargs):
        print("plant_simulation")        
        # simulates the current 
        disturbance_src = kwargs.pop('disturbance_src', 'process_noise')
        # options: process_noise --> gaussian noise added to all states
        #          parameter noise --> noise added to the uncertain model parameters
        #                               or different ones
        #          input noise --> noise added to the inputs/controls
        # not exhaustive list, can be adapted/extended as one wishes
        # combination of all the above with noise in the initial point supported
        initial_disturbance = kwargs.pop('initial_disturbance', {(x,j):0.0 for x in self.states for j in self.state_vars[x]})
        parameter_disturbance = kwargs.pop('parameter_disturbance', {})
        state_disturbance = kwargs.pop('state_disturbance', {})
        input_disturbance = kwargs.pop('input_disturbance',{})
                
        #  generate the disturbance according to specified scenario
        state_noise = {}
        input_noise = {}
        if disturbance_src == 'process_noise':
            if state_disturbance != {}: 
                for key in state_disturbance:
                    state_noise[key] = np.random.normal(loc=0.0, scale=state_disturbance[key])
                    # potentially implement truncation at 2 sigma
            else:
                for x in self.states: 
                    for j in self.state_vars[x]:
                        state_noise[(x,j)] = 0.0   
            
            for u in self.u:
                input_noise[u] = 0.0
                
        elif disturbance_src == 'parameter_noise':
            for p in parameter_disturbance:
                disturbed_parameter = getattr(self.plant_simulation_model, p[0])
                if first_call:
                    self.nominal_parameter_values[p] = disturbed_parameter[p[1]].value
                
                if (self.iterations-1)%parameter_disturbance[p][1] == 0:
                    disturbed_parameter[p[1]].value = self.nominal_parameter_values[p] * (1 + np.random.normal(loc=0.0, scale=parameter_disturbance[p][0]))                 
                    
            for x in self.states:
                for j in self.state_vars[x]:
                    state_noise[(x,j)] = 0.0
                    
            for u in self.u:
                input_noise[u] = 0.0
        elif disturbance_src == 'input_disturbance': # input noise only
            for x in self.states:
                for j in self.state_vars[x]:
                    state_noise[(x,j)] = 0.0
            for u in self.u:
                input_noise[u] = np.random.normal(loc=0.0, scale=input_disturbance[u])
        else:
            print('NO DISTURBANCE SCENARIO SPECIFIED, NO NOISE ADDED ANYWHERE')                     
            for x in self.states:
                for j in self.state_vars[x]:
                        state_noise[(x,j)] = 0.0
            for u in self.u:
                input_noise[u] = 0.0
                
        if first_call: 
            for x in self.states:
                    xic = getattr(self.plant_simulation_model,x+'_ic')
                    x_var = getattr(self.plant_simulation_model,x)
                    for j in self.state_vars[x]:
                        if j == ():
                            xic.value = xic.value * (1 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j)]))
                            aux = xic.value
                        else:
                            xic[j].value = xic[j].value * (1 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j)]))
                            aux = xic[j].value
                        # just for initialization purposes for the simulation
                        for k in range(0,self.ncp_t+1): 
                            x_var[(1,k)+j].value = aux
                            
            # initialization of the simulation (initial guess)               
            for var in self.plant_simulation_model.component_objects(Var, active=True):
                var_ref = getattr(self.recipe_optimization_model, var.name)
                for key in var.index_set():
                    if var[key].fixed:
                        break
                    else:
                        var[key].value = var_ref[key].value
        else:                
            # initialization of simulation
            for var in self.plant_simulation_model.component_objects(Var, active=True):
                for key in var.index_set():
                    if var[key].fixed:
                        break
                    elif key == None or type(key) == int:
                        continue
                    elif type(key) == tuple and type(key[0])==int:
                        if len(key) > 2:
                            var[key] = var[(1,self.ncp_t)+key[2:]].value 
                        else:
                            var[key] = var[(1,self.ncp_t)].value
                            
            # initialization of state trajectories
            # adding state noise if specified
            for x in self.states:
                xic = getattr(self.plant_simulation_model,x+'_ic')
                x_var = getattr(self.plant_simulation_model,x)
                for j in self.state_vars[x]:
                    if j == (): 
                        xic.value = x_var[(1,self.ncp_t)+j].value * (1.0 + state_noise[(x,j)])#+ noise # again tailored to RADAU nodes
                    else:
                        xic[j].value = x_var[(1,self.ncp_t)+j].value * (1.0 + state_noise[(x,j)])# + noise # again tailored to RADAU nodes
                    # for initialization: take constant values + leave time invariant values as is!
                    for k in range(0,self.ncp_t+1):
                        x_var[(1,k)+j].value = x_var[(1,self.ncp_t)+j].value
 
        # result is the previous olnmpc solution 
        #    --> therefore provides information about the sampling interval
        # 1 element model for forward simulation
        self.plant_simulation_model.tf = result['tf', None]
        self.plant_simulation_model.tf.fixed = True
                
        # FIX(!!) the controls, path constraints are deactivated by default
        for u in self.u:
            control = getattr(self.plant_simulation_model,u)
            control[1].value = result[u,1]*(1.0+input_noise[u])
            control[1].fixed = True
            self.plant_simulation_model.equalize_u(direction="u_to_r")
               
        # probably redundant
        self.plant_simulation_model.clear_aux_bounds()
        
        # solve statement
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 3000
        
        out = ip.solve(self.plant_simulation_model, tee=True)
        
        # check if converged otw. run again and hope for numerical issues
        if [str(out.solver.status), str(out.solver.termination_condition)] != ['ok','optimal']:
            out = ip.solve(self.plant_simulation_model, tee = True)
        
        self.plant_trajectory[self.iterations,'solstat'] = [str(out.solver.status), str(out.solver.termination_condition)]
        self.plant_trajectory[self.iterations,'tf'] = self.plant_simulation_model.tf.value
        
        # safe results of the plant_trajectory dictionary {number of iteration, (x,j): value}
        for u in self.u:
            control = getattr(self.plant_simulation_model,u)
            self.plant_trajectory[self.iterations,u] = control[1].value #(control valid between 0 and tf)
        for x in self.states:
            xvar = getattr(self.plant_simulation_model, x)
            for j in self.state_vars[x]:
                    self.plant_trajectory[self.iterations,(x,j)] = xvar[(1,3)+j].value 
                    # setting the current real value w/o measurement noise
                    self.curr_rstate[(x,j)] = xvar[(1,3)+j].value 
      
        # monitor path constraints for results
        if self.path_constraints != []:
            for pc_name in self.path_constraints:
                pc_var = getattr(self.plant_simulation_model, pc_name)
                for index in pc_var.index_set():
                    if self.multimodel or self.linapprox:
                        self.pc_trajectory[(pc_name,(self.iterations,index[1:-1]))] = pc_var[index].value
                    else:
                        self.pc_trajectory[(pc_name,(self.iterations,index[1:]))] = pc_var[index].value
            
            for cp in range(self.ncp_t+1):
                self.pc_trajectory[('tf',(self.iterations,cp))] = self.plant_simulation_model.tau_i_t[cp]*self.plant_simulation_model.tf.value

    def set_predicted_state(self,m):
        self.predicted_state = {}
        for _var in m.component_objects(Var, active=True):
            for _key in _var.index_set():
                try:
                    self.predicted_state[(_var.name,_key)] = _var[_key].value
                except KeyError:
                    continue
    
    def forward_simulation(self):
        print("forward_simulation")
        # simulates forward from the current measured or (!) esimated state
        # using the sensitivity based updated control inputs
        
        # IMPORTANT:
        # --> before calling forward_simulation() need to call dot_sens to perform the update
        #
                
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
            
        self.forward_simulation_model.tf = self.olnmpc.tf.value
        self.forward_simulation_model.tf.fixed = True  
        
        # Update adpated params
        if self.adapt_params and self.iterations > 1:
            for index in self.curr_epars:
                p = getattr(self.forward_simulation_model, index[0])
                p[index[1]].value = self.curr_epars[index]

        # general initialization
        # not super sophisticated but works
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
        for x in self.states:
            xic = getattr(self.forward_simulation_model,x+'_ic')
            xvar = getattr(self.forward_simulation_model,x)
            for j in self.state_vars[x]:
                if j == (): 
                    xic.value = current_state_info[(x,j)]
                else:
                    xic[j].value = current_state_info[(x,j)]
                # just initialization heuristic --> change if you want to
                for k in range(0,self.ncp_t+1):# use only 
                    xvar[(1,k)+j].value = current_state_info[(x,j)]
                    
        # set and fix control as provided by olnmpc/advanced step update     
        for u in self.u:
            control = getattr(self.forward_simulation_model,u)
            control[1].value = current_control_info[u]           
            control[1].fixed = True
            self.forward_simulation_model.equalize_u(direction="u_to_r")
        
        self.forward_simulation_model.clear_aux_bounds()
        # solve statement
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 3000
  
        out = ip.solve(self.forward_simulation_model, tee=True)
        self.simulation_trajectory[self.iterations,'obj_fun'] = 0.0
        
        if [str(out.solver.status), str(out.solver.termination_condition)] != ['ok','optimal']:
            self.forward_simulation_model.clear_aux_bounds()
            out = ip.solve(self.forward_simulation_model, tee = True)
            self.simulation_trajectory[self.iterations,'obj_fun'] = 0.0
            
        self.simulation_trajectory[self.iterations,'solstat'] = [str(out.solver.status), str(out.solver.termination_condition)]
        
        # save the simulated state as current predicted state
        # implicit assumption of RADAU nodes
        for x in self.states:
            xvar = getattr(self.forward_simulation_model, x)
            for j in self.state_vars[x]:
                    self.curr_pstate[(x,j)] = xvar[(1,self.ncp_t)+j].value  
                      
                
    def recipe_optimization(self):
        self.nfe_t_0 = self.nfe_t # set self.nfe_0 to keep track of the length of the reference trajectory
        self.generate_state_index_dictionary()
        self.recipe_optimization_model = self.d_mod(self.nfe_t, self.ncp_t)
        self.recipe_optimization_model.initialize_element_by_element()
        self.recipe_optimization_model.create_bounds()
        self.recipe_optimization_model.create_output_relations()
        self.recipe_optimization_model.clear_aux_bounds()
        #self.recipe_optimization_model.aux.fixed = True
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["max_iter"] = 5000
        results = ip.solve(self.recipe_optimization_model, tee=True, symbolic_solver_labels=True, report_timing=True)
        if not(str(results.solver.status) == 'ok' and str(results.solver.termination_condition)) == 'optimal':
            print('Recipe Optimization failed!')
            sys.exit()
            
        
        self.nmpc_trajectory[1,'tf'] = self.recipe_optimization_model.tf.value # start at 0.0
        for u in self.u:
            control = getattr(self.recipe_optimization_model,u)
            self.nmpc_trajectory[1,u] = control[1].value # control input bewtween time step i-1 and i
            self.curr_u[u] = control[1].value
        
        # directly apply state as predicted state --> forward simulation would reproduce the exact same result since no additional measurement is known
        for x in self.states:
            xvar = getattr(self.recipe_optimization_model,x)
            for j in self.state_vars[x]:
                self.curr_pstate[(x,j)] = xvar[(1,self.ncp_t)+j].value
                self.initial_values[(x,j)] = xvar[(1,self.ncp_t)+j].value
        
    def get_tf(self):
        return self.recipe_optimization_model.tf.value
    
    def set_reference_state_trajectory(self,input_trajectory):
        # input_trajectory = {('state',(fe,cp,j)):value}
        self.reference_state_trajectory = input_trajectory

    def get_state_trajectory(self,d_mod):
        output = {}
        for x in self.states:
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
        
    def set_reference_control_trajectory(self,control_trajectory):
        # input_trajectory = {('state',(fe,j)):value} ---> this format
        self.reference_control_trajectory = control_trajectory
        
    def get_control_trajectory(self,d_mod):
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
     
    def generate_state_index_dictionary(self):
        # generates a dictionary = {'name':[indices except fe,cp] if only 1 add. index (j,) if none ()}
        for x in self.states:
            self.state_vars[x] = []
            xv = getattr(self.forward_simulation_model, x)
            for j in xv.keys():
                if j[1] == 0:
                    if type(j[2:]) == tuple:
                        self.state_vars[x].append(j[2:])
                    else:
                        self.state_vars[x].append((j[2:],))
                else:
                    continue
         
    def create_enmpc(self):
        self.olnmpc = self.d_mod(self.nfe_t,self.ncp_t)
        self.olnmpc.name = "olnmpc (Open-Loop eNMPC)"
        self.olnmpc.create_bounds()
        self.olnmpc.clear_aux_bounds()
        
    def create_nmpc(self):
        if self.multimodel:
            self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t, n_s = self.n_s)
            # if scenario tree is not updated self.st = {}
            for index in self.st: #scenario tree: self.st = {'parname','key',l:value} with l being the scenario
                p = getattr(self.olnmpc, 'p_' + index[0])
                p[index[1],index[2]] = self.st[index]
        elif self.linapprox:
            self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t)
        else:    
            self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t)
            
        self.olnmpc.name = "olnmpc (Open-Loop NMPC)"
        self.olnmpc.create_bounds()
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
            for x in self.states:
                x_var = getattr(self.olnmpc, x)  #: State
                for j in self.state_vars[x]:
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
        # tracking terms according to xmpc_l/umpc_l
        self.olnmpc.xmpcS_nmpc = Set(initialize=[i for i in range(0, len(self.xmpc_l[1]))])
        self.olnmpc.umpcS_nmpc = Set(initialize=[i for i in range(0, len(self.umpc_l[1]))])
        # A: The reference trajectory
        self.olnmpc.xmpc_ref_nmpc = Param(self.olnmpc.fe_t, self.olnmpc.xmpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.umpc_ref_nmpc = Param(self.olnmpc.fe_t, self.olnmpc.umpcS_nmpc, initialize=0.0, mutable=True)
        # B: Weights for the different states (for x (Q) and u (R))
        self.olnmpc.Q_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=1.0, mutable=True)  
        self.olnmpc.R_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=1.0, mutable=True) 
        # C: Weights for the different finite elements (time dependence) (for x (Q) and u (R))
        self.olnmpc.Q_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True) # 1e-4
        self.olnmpc.R_w_nmpc = Param(self.olnmpc.fe_t, initialize=0.0, mutable=True) # 1e2
        
        # generate the expressions for the objective function
        
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
        if self.multimodel or self.linapprox: # for those two model types additionally scenarios need to be accounted for
            self.olnmpc.objfun_nmpc = Objective(expr = (self.olnmpc.tf + self.olnmpc.rho*sum(sum(self.olnmpc.eps[i,s] for i in self.olnmpc.epc) for s in self.olnmpc.s)+self.olnmpc.xQ_expr_nmpc + self.olnmpc.xR_expr_nmpc))        
        else:
            self.olnmpc.objfun_nmpc = Objective(expr = (self.olnmpc.tf + self.olnmpc.rho*sum(self.olnmpc.eps[i] for i in self.olnmpc.epc)+self.olnmpc.xQ_expr_nmpc + self.olnmpc.xR_expr_nmpc))
        
    def load_reference_trajectories(self):
        # assign values of the reference trajectory to parameters 
        # self.olnmpc.xmpc_ref_nmpc and self.umpc_ref_nmpc
        
        # reference trajectory used to assign delta
        # tracking terms shifted by every iteration (therefore self.iterations+1)
        # first time this should be called is for i = 1 
        # --> only tracking terms for k >= 2 relevant
        for x in self.states:
            for j in self.state_vars[x]:
                for fe in range(1, self.nfe_t+1):
                    try:
                        self.olnmpc.xmpc_ref_nmpc[fe,self.xmpc_key[(x,j)]] = self.reference_state_trajectory[x,(fe+self.iterations,self.ncp_t)+j]
                        try:
                            self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0/(self.reference_state_trajectory[x,(fe+self.iterations,self.ncp_t)+j] + 0.01)**2
                        except ZeroDivisionError:
                            self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0
                    except KeyError:
                        self.olnmpc.xmpc_ref_nmpc[fe,self.xmpc_key[(x,j)]] = self.reference_state_trajectory[x,(self.nfe_t_0,self.ncp_t)+j]
                        try:
                            self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0/(self.reference_state_trajectory[x,(self.nfe_t_0,self.ncp_t)+j] + 0.01)**2
                        except ZeroDivisionError:
                            self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0
        
        for u in self.u:
            #control = getattr(self.olnmpc, u)
            for fe in range(1, self.nfe_t+1):
                try:
                    self.olnmpc.umpc_ref_nmpc[fe,self.umpc_key[u]] = self.reference_control_trajectory[u,fe+self.iterations]
                    try:
                        self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(self.reference_control_trajectory[u,fe+self.iterations] + 0.01)**2
                        #self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(control[1].ub-control[1].lb)**2
                    except ZeroDivisionError:
                        self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0
                except KeyError:
                    self.olnmpc.umpc_ref_nmpc[fe,self.umpc_key[u]] = self.reference_control_trajectory[u,self.nfe_t_0]
                    try:
                        self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(self.reference_control_trajectory[u,self.nfe_t_0] + 0.01)**2
                        #self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(control[1].ub-control[1].lb)**2
                    except ZeroDivisionError:
                        self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0
        
        
    def store_results(self,m):
        # store the results of an entire optimization problem into one dictionary
        # output = {'name',(key):value}
        output = {}
        for _var in m.component_objects(Var, active=True):
            for _key in _var.index_set():
                try:
                    output[(_var.name,_key)] = _var[_key].value
                except KeyError:
                    continue
        return output       

    def cycle_nmpc(self,initialguess):
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
                    if self.multimodel or self.linapprox:
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
        
        # adapt parameters obtained from on-line estimation
        if self.adapt_params and self.iterations > 1:
            for index in self.curr_epars:
                p = getattr(self.olnmpc, index[0])
                p[index[1]].value = self.curr_epars[index]
        

        # set initial value parameters in model olnmpc
        # set according to predicted state by forward simulation
        # a) values will not be overwritten in case advanced step is used
        # b) values will be overwritten by call of add_noise() in case advanced step is not used
        for x in self.states:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.state_vars[x]:
                if not(j == ()):
                    xic[j].value = self.curr_pstate[(x,j)]
                else:
                    xic.value = self.curr_pstate[(x,j)]
        
    def reinit(self, initialguess):
    # initialize new problem with the shifted old one
        for _var in self.olnmpc.component_objects(Var, active=True):
            for _key in _var.index_set():
                if not(_key == None): # if the variable is time invariant scalar skip this 
                    if type(_key) == tuple: # for variables that are indexed not only by number of finite element               
                        shifted_element = (_key[0] + 1,)   # shift finite element by 1
                        aux_key = (_var.name,shifted_element + _key[1:]) # adjust key
                    else: # for variables that are only indexed by number of finite element
                        shifted_element = _key + 1      # shift finite element by 1
                        aux_key = (_var.name,shifted_element)
                else:
                    aux_key = (_var.name,_key)
                try:
                    _var[_key].value = initialguess[aux_key]
                except KeyError: # last element will be doubled since exception is thrown if _key[0] == self.nfe_t
                    continue  
                
        for x in self.states:
            for j in self.state_vars[x]:
                self.initial_values[(x,j)] = initialguess[(x,(2,0)+j)] 

        # set initial value parameters in mode olnmpc
        for x in self.states:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.state_vars[x]:
                if not(j == ()):
                    xic[j].value = self.initial_values[(x,j)]
                else:
                    xic.value = self.initial_values[(x,j)]
        
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
        for x in self.states:
            s = getattr(self.noisy_model, x)  #: state
            xicc = getattr(self.noisy_model, x + "_icc")
            xicc.deactivate()
            for j in self.state_vars[x]:
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
        for x in self.states:
            s = getattr(self.noisy_model, x)  #: state
            xic = getattr(self.noisy_model, x + "_ic")
            for j in self.state_vars[x]:
                if not(j == ()):
                    expr = s[(1, 0) + j] == xic[j] + self.noisy_model.w_pnoisy[self.xp_key[(x,j)]] # changed(!)
                else:
                    expr = s[(1, 0) + j] == xic + self.noisy_model.w_pnoisy[self.xp_key[(x,j)]] 
                self.noisy_model.ics_noisy.add(expr)
                
    def add_noise(self,var_dict):
        print("add_noise")
        # set the time horizon
        self.noisy_model.tf = self.recipe_optimization_model.tf.value # just a dummy value in order to prohibit tf to go to 0
        # to account for possibly inconsistent initial values
        # solve auxilliary problems with only 1 finite element  
        for x in self.states:
            xic = getattr(self.noisy_model, x + '_ic')
            for j in self.state_vars[x]:
                if j == ():
                    #xic[j].value = self.initial_values[(x,j)]
                    xic.value = self.curr_rstate[(x,j)]
                else:
                    #xic.value = self.initial_values[(x,j)]
                    xic[j].value = self.curr_rstate[(x,j)] 
        
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

        results = ip.solve(self.noisy_model, tee=True)
        
        self.nmpc_trajectory[self.iterations, 'solstat_noisy'] = [str(results.solver.status),str(results.solver.termination_condition)]
        self.nmpc_trajectory[self.iterations, 'obj_noisy'] = value(self.noisy_model.obj_fun_noisy.expr)
        # save the new consistent initial conditions 
        for key in var_dict:
            vni = self.xp_key[key]
            self.curr_rstate[key] = self.noisy_model.w_pnoisy[vni].value + self.curr_rstate[key] # from here on it is a noisy real state
            
        # Set new (noisy) initial conditions in model self.olnmpc
        # + save in real_trajectory dictionary
        for x in self.states:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.state_vars[x]:
                if not(j == ()):
                    #xic[j].value = self.initial_values[(x,j)] # set noisy initial value                   
                    xic[j].value = self.curr_rstate[(x,j)]                   
                else:
                    #xic.value = self.initial_values[(x,j)] # set noisy initial value 
                    xic.value = self.curr_rstate[(x,j)]                    
        # for trouble shooting to check how much the consistent initial conditions deviates from the completely random one
        for key in var_dict:    
            vni = self.xp_key[key]
            self.nmpc_trajectory[self.iterations,'noise',key] = self.noisy_model.w_pnoisy[vni].value - self.noisy_model.w_ref[vni].value
        
        
            
    def solve_olnmpc(self):
        print("solve_olnmpc")
        print('#'*20 + ' ' + str(self.iterations) + ' ' + '#'*20)
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        ip.options["tol"] = 1e-8
        ip.options["max_iter"] = 3000
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
        self.olnmpc.write_nl()
        if not(str(results.solver.status) == 'ok' and str(results.solver.termination_condition) == 'optimal'):
            print('olnmpc failed to converge')
            
        # save relevant results
        # self.iterations + 1 holds always if solve_olnmpc called at the correct time
        self.nmpc_trajectory[self.iterations,'solstat'] = [str(results.solver.status),str(results.solver.termination_condition)]
        self.nmpc_trajectory[self.iterations+1,'tf'] = self.nmpc_trajectory[self.iterations,'tf'] + self.olnmpc.tf.value
        self.nmpc_trajectory[self.iterations,'eps'] = [self.olnmpc.eps[i].value for i in self.olnmpc.eps.index_set()]  
        if self.obj_type == 'economic':
            self.nmpc_trajectory[self.iterations,'obj_value'] = value(self.olnmpc.eobj)
        else:
            self.nmpc_trajectory[self.iterations,'obj_value'] = value(self.olnmpc.objfun_nmpc)
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.nmpc_trajectory[self.iterations+1,u] = control[1].value # control input between timestep i-1 and i
        
        # initial state is saved to keep track of how good the estimates are
        # here only self.iteration since its the beginning of the olnmpc run
        for x in self.states:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.state_vars[x]:
                if j == ():
                    self.nmpc_trajectory[self.iterations,(x,j)] = xic.value   
                else:
                    self.nmpc_trajectory[self.iterations,(x,j)] = xic[j].value
        
        # save the control result as current control input
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.curr_u[u] = control[1].value
    
        
    def solve_olrnmpc(self,cons,**kwargs):
        # open-loop robust nonlinear model predictive control problem with linear approximation of the lower level program
        print("solve_olrnmpc")
        print('#'*20 + ' ' + str(self.iterations) + ' ' + '#'*20)

        # specify inputs
            # cons: list of the 'name's of the slack variables 's_name'
    
        # set algorithmic options
        iterlim = kwargs.pop('iterlim',10)  
        eps = kwargs.pop('eps',0.0)
        
        ip = SolverFactory('asl:ipopt')
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["max_iter"] = 5000
        ip.options["tol"] = 1e-8
        ip.options["linear_solver"] = "ma57"
        
        k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
        k_aug.options["compute_dsdp"] = ""
        iters = 0
        converged = False
        
        # initialize everything
        backoff = {}
        for i in cons:
            backoff_var = getattr(self.olnmpc,'xi_'+i)
            for index in backoff_var.index_set():
                try:
                    backoff[('s_'+i,index)] = 0.0
                    backoff_var[index].value = 0.0
                except KeyError:
                    continue
        n_p = 0
        for p in self.p_noisy:
            for key in self.p_noisy[p]:
                n_p += 1
        
        print('iterations:')
        while (iters < iterlim and not(converged)):
            print(' '*7 + str(iters))
            # solve optimization problem
            self.olnmpc.create_bounds()
            self.olnmpc.clear_aux_bounds()
            for u in self.u:
                u_var = getattr(self.olnmpc,u)
                u_var.unfix()
            self.olnmpc.tf.unfix()
            self.olnmpc.eps.unfix()
            
            for i in cons:
                slack = getattr(self.olnmpc, 's_'+i)
                for index in slack.index_set():
                    slack[index].setlb(0)
            nlp_results = ip.solve(self.olnmpc, tee=False)

            if [str(nlp_results.solver.status),str(nlp_results.solver.termination_condition)] != ['ok','optimal']:
                self.olnmpc.A.pprint()
                self.olnmpc.write_nl()
                self.olnmpc.troubleshooting()
                sys.exit()
            
            # check whether optimal control problem feasible
            flag = False
            for index in self.olnmpc.eps.index_set():
                if self.olnmpc.eps[index].value > 1e-3:
                    flag = True
            if flag:
                break
            
            self.olnmpc.eps.fix()
            for u in self.u:
                u_var = getattr(self.olnmpc,u)
                u_var.fix()
            self.olnmpc.tf.fix()
            self.olnmpc.clear_all_bounds()
                
            # compute sensitivities
            self.olnmpc.ipopt_zL_in.update(self.olnmpc.ipopt_zL_out)
            self.olnmpc.ipopt_zU_in.update(self.olnmpc.ipopt_zU_out)
        
            if iters == 0:
                self.olnmpc.var_order = Suffix(direction=Suffix.EXPORT)
                self.olnmpc.dcdp = Suffix(direction=Suffix.EXPORT)
                i = 1
                reverse_dict_pars ={}
                for p in self.p_noisy:
                    for key in self.p_noisy[p]:
                        dummy = 'dummy_constraint_p_' + p + '_' + key
                        dummy_con = getattr(self.olnmpc, dummy)
                        for index in dummy_con.index_set():
                            self.olnmpc.dcdp.set_value(dummy_con[index], i)
                            reverse_dict_pars[i] = (p,key)
                            i += 1
            
                i = 1
                reverse_dict_cons = {}
                for k in cons:
                    s = getattr(self.olnmpc, 's_'+k)
                    for index in s.index_set():
                        if not(s[index].stale):
                            self.olnmpc.var_order.set_value(s[index], i)
                            reverse_dict_cons[i] = ('s_'+ k,index)
                            i += 1
                    
            k_aug.solve(self.olnmpc, tee=False)
            
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
                    sens[key] = self.alpha*sens[key]
            elif type(self.alpha) == dict:
                # case b): hyperrectangle
                for key in sens:
                    sens[key] = self.alpha[key[1]]*sens[key]
            elif type(self.alpha) == tuple and self.alpha[1] == 'adapted':
                # case c): weighting_matrix --> rectangle is tilted
                # self made matrix multiplication to ensure the rows and cols match...
                if type(self.weighting_matrix) != dict:
                    for key in sens:
                        #key[1] = p # the exact same index for self.PI_indices
                        #self.PI_indices[key[1]] returns index for corresponding parameter in _PI
                        s = key[0] 
                        m = self.PI_indices[key[1]]
                        sens[key] = sum(sens[s,p]*self.weighting_matrix[m][self.PI_indices[p]] for p in self.PI_indices)
                else:
                    for key in sens:
                        sens[key] = self.alpha[0][key[1]]*sens[key]
            else:
                print('ERROR: WRONG SPECIFICATION OF CONFIDENCE REGION')
                sys.exit()
            # convergence check and update    
            converged = True
            for i in cons:
                backoff_var = getattr(self.olnmpc,'xi_'+i)
                for index in backoff_var.index_set():
                    try:
                        # computes the 1-norm of the respective part of the sensitivity matrix
                        new_backoff = sum(abs(sens[(('s_'+i,index),reverse_dict_pars[k])]) for k in range(1,n_p+1))
                        # update backoff margins 
                        old_backoff = backoff[('s_'+i,index)]
                        if backoff[('s_'+i,index)] - new_backoff < 0:
                            backoff[('s_'+i,index)] = new_backoff
                            backoff_var[index].value = new_backoff
                            # check convergence
                            if  old_backoff - new_backoff <= -eps:
                                converged = False
                        else:
                            continue
                    except KeyError:
                        continue
                    
            iters += 1
            
        self.olnmpc.create_bounds()
        self.olnmpc.clear_aux_bounds()
        for u in self.u:
            u_var = getattr(self.olnmpc,u)
            u_var.unfix()
        self.olnmpc.tf.unfix()
        
        for i in cons:
            slack = getattr(self.olnmpc, 's_'+i)
            for index in slack.index_set():
                slack[index].setlb(0)
        ip.solve(self.olnmpc,tee=True)
            
           
        # save converged results 
        self.nmpc_trajectory[self.iterations,'solstat'] = [str(nlp_results.solver.status),str(nlp_results.solver.termination_condition)]
        self.nmpc_trajectory[self.iterations+1,'tf'] = self.nmpc_trajectory[self.iterations,'tf'] + self.olnmpc.tf.value
        self.nmpc_trajectory[self.iterations,'eps'] = [self.olnmpc.eps[i].value for i in self.olnmpc.eps.index_set()]  
        if self.obj_type == 'economic':
            self.nmpc_trajectory[self.iterations,'obj_value'] = value(self.olnmpc.eobj)
        else:
            self.nmpc_trajectory[self.iterations,'obj_value'] = value(self.olnmpc.objfun_nmpc)
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.nmpc_trajectory[self.iterations+1,u] = control[1].value # control input between timestep i-1 and i
        
        for x in self.states:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.state_vars[x]:
                if j == ():
                    self.nmpc_trajectory[self.iterations,(x,j)] = xic.value   
                else:
                    self.nmpc_trajectory[self.iterations,(x,j)] = xic[j].value
        
        # save the control result as current control input
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.curr_u[u] = control[1].value
        
    
        self.olnmpc.write_nl()
        
        
    def cycle_iterations(self):
        self.iterations += 1
        self.nfe_mhe += 1
        
###########################################################################                
###########################################################################                
###########################################################################               
###########################################################################        

    def initialize_olnmpc(self, ref, src_kind, **kwargs):
        # The reference is always a model
        # The source of the state might be different
        # The source might be a predicted-state from forward simulation
        """Initializes the olnmpc from a reference state, loads the state into the olnmpc
        Args
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model
            fe (int): Source fe
            src_kind (str): the kind of source
        Returns:
            """
        fe = kwargs.pop("fe", 1)
        self.journalizer("I", self._c_it, "initialize_olnmpc", "Attempting to initialize olnmpc")
        self.journalizer("I", self._c_it, "initialize_olnmpc", "src_kind=" + src_kind)
        # self.load_init_state_nmpc(src_kind="mod", ref=ref, fe=1, cp=self.ncp_t)

        if src_kind == "real":
            self.load_init_state_nmpc(src_kind="dict", state_dict="real")
        elif src_kind == "estimated":
            self.load_init_state_nmpc(src_kind="dict", state_dict="estimated")
        elif src_kind == "predicted":
            self.load_init_state_nmpc(src_kind="dict", state_dict="predicted")
        else:
            self.journalizer("E", self._c_it, "initialize_olnmpc", "SRC not given")
            sys.exit()
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.create_bounds()
        #: Load current solution
        self.load_d_d(ref, dum, fe, fe_src="s")  #: This is supossed to work
        for u in self.u:  #: Initialize controls dummy model
            cv_dum = getattr(dum, u)
            cv_ref = getattr(ref, u)
            for i in cv_dum.keys():
                cv_dum[i].value = value(cv_ref[fe])
        #: Patching of finite elements
        k_notopt = 0
        for finite_elem in range(1, self.nfe_t + 1):
            dum.name = "Dummy I " + str(finite_elem)
            if finite_elem == 1:
                if src_kind == "predicted":
                    self.load_init_state_gen(dum, src_kind="dict", state_dict="predicted")
                elif src_kind == "estimated":
                    self.load_init_state_gen(dum, src_kind="dict", state_dict="estimated")
                elif src_kind == "real":
                    self.load_init_state_gen(dum, src_kind="dict", state_dict="real")
                else:
                    self.journalizer("E", self._c_it, "initialize_olnmpc", "SRC not given")
                    sys.exit()
            else:
                self.load_init_state_gen(dum, src_kind="mod", ref=dum, fe=1)

            tst = self.solve_d(dum,
                               o_tee=False,
                               tol=1e-06,
                               stop_if_nopt=False,
                               output_file="dummy_ip.log")
            if tst != 0:
                self.journalizer("W", self._c_it, "initialize_olnmpc", "non-optimal dummy")
                self.solve_d(dum,
                             o_tee=True,
                             tol=1e-03,
                             stop_if_nopt=False,
                             output_file="dummy_ip.log")
                k_notopt += 1
            #: Patch
            self.load_d_d(dum, self.olnmpc, finite_elem)

            for u in self.u:
                cv_nmpc = getattr(self.olnmpc, u)  #: set controls for open-loop nmpc
                cv_dum = getattr(dum, u)
                # works only for fe_t index
                cv_nmpc[finite_elem].set_value(value(cv_dum[1]))
        self.journalizer("I", self._c_it, "initialize_olnmpc", "Done, k_notopt " + str(k_notopt))

    def load_init_state_nmpc(self, src_kind, **kwargs):
        """Loads ref state for set-point
        Args:
            src_kind (str): the kind of source
            **kwargs: Arbitrary keyword arguments.
        Returns:
            None
        Keyword Args:
            src_kind (str) : if == mod use reference model, otw use the internal dictionary
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model (default d1)
            fe (int): The required finite element
            cp (int): The required collocation point
        """
        # src_kind = kwargs.pop("src_kind", "mod")
        self.journalizer("I", self._c_it, "load_init_state_nmpc", "Load State to nmpc src_kind=" + src_kind)
        ref = kwargs.pop("ref", None)
        fe = kwargs.pop("fe", self.nfe_t)
        cp = kwargs.pop("cp", self.ncp_t)
        if src_kind == "mod":
            if not ref:
                self.journalizer("W", self._c_it, "load_init_state_nmpc", "No model was given")
                self.journalizer("W", self._c_it, "load_init_state_nmpc", "No update on state performed")
                sys.exit()
            for x in self.states:
                xic = getattr(self.olnmpc, x + "_ic")
                xvar = getattr(self.olnmpc, x)
                xsrc = getattr(ref, x)
                for j in self.state_vars[x]:
                    xic[j].value = value(xsrc[(fe, cp) + j])
                    xvar[(1, 0) + j].set_value(value(xsrc[(fe, cp) + j]))
        else:
            state_dict = kwargs.pop("state_dict", None)
            if state_dict == "real":  #: Load from the real state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_rstate[(x, j)]
                        xvar[(1, 0) + j].set_value(self.curr_rstate[(x, j)])
            elif state_dict == "estimated":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_estate[(x, j)]
                        xvar[(1, 0) + j].set_value(self.curr_estate[(x, j)])
            elif state_dict == "predicted":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_pstate[(x, j)]
                        xvar[(1, 0) + j].set_value(self.curr_pstate[(x, j)])
            else:
                self.journalizer("W", self._c_it, "load_init_state_nmpc", "No dict w/state was specified")
                self.journalizer("W", self._c_it, "load_init_state_nmpc", "No update on state performed")
                sys.exit()

    def compute_QR_nmpc(self, src="plant", n=-1, **kwargs):
        """Using the current state & control targets, computes the Qk and Rk matrices (diagonal)
        Args:
            src (str): The source of the update (default mhe) (mhe or plant)
            n (int): The exponent of the weight"""
        check_values = kwargs.pop("check_values", False)
        if check_values:
            max_w_value = kwargs.pop("max_w_value", 1e+06)
            min_w_value = kwargs.pop("min_w_value", 0.0)
        self.update_targets_nmpc()
        if src == "mhe":
            for x in self.states:
                for j in self.state_vars[x]:
                    k = self.xmpc_key[(x, j)]
                    self.olnmpc.Q_nmpc[k].value = abs(self.curr_estate[(x, j)] - self.curr_state_target[(x, j)])**n
                    self.olnmpc.xmpc_ref_nmpc[k].value = self.curr_state_target[(x, j)]
        elif src == "plant":
            for x in self.states:
                for j in self.state_vars[x]:
                    k = self.xmpc_key[(x, j)]
                    self.olnmpc.Q_nmpc[k].value = abs(self.curr_rstate[(x, j)] - self.curr_state_target[(x, j)])**n
                    self.olnmpc.xmpc_ref_nmpc[k].value = self.curr_state_target[(x, j)]
        k = 0
        for u in self.u:
            self.olnmpc.R_nmpc[k].value = abs(self.curr_u[u] - self.curr_u_target[u])**n
            self.olnmpc.umpc_ref_nmpc[k].value = self.curr_u_target[u]
            k += 1
        if check_values:
            for k in self.olnmpc.xmpcS_nmpc:
                if value(self.olnmpc.Q_nmpc[k]) < min_w_value:
                    self.olnmpc.Q_nmpc[k].value = min_w_value
                if value(self.olnmpc.Q_nmpc[k]) > max_w_value:
                    self.olnmpc.Q_nmpc[k].value = max_w_value
            k = 0
            for u in self.u:
                if value(self.olnmpc.R_nmpc[k]) < min_w_value:
                    self.olnmpc.R_nmpc[k].value = min_w_value
                if value(self.olnmpc.R_nmpc[k]) > max_w_value:
                    self.olnmpc.R_nmpc[k].value = max_w_value
                k += 1

    def new_weights_olnmpc(self, state_weight, control_weight):
        if type(state_weight) == float:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[fe].value = state_weight
        elif type(state_weight) == dict:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[fe].value = state_weight[fe]

        if type(control_weight) == float:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.R_w_nmpc[fe].value = control_weight
        elif type(control_weight) == dict:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.R_w_nmpc[fe].value = control_weight[fe]

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

    def create_nmpc_sIpopt_suffixes(self):
        m = self.olnmpc
        m.sens_state_0 = Suffix(direction=Suffix.EXPORT) # enumerates parameters that will be perturbed --> unique mapping number --> Var(actual parameter)
        m.sens_state_1 = Suffix(direction=Suffix.EXPORT) # enumerates parameters that are pertured --> mapping has to be equivalent to sens_state_0
        m.sens_state_value_1 = Suffix(direction=Suffix.EXPORT) # holds the perturbed parameter values, has to be set for the same variables as sens_state_1 is
        m.sens_sol_state_1 = Suffix(direction=Suffix.IMPORT) # holds the updated variables and lagrange multipliers for inequalities (and equalities (would be relatively meaninglessS?)) 
        m.sens_init_constr = Suffix(direction=Suffix.EXPORT) # flag constraint that are artificial Var(actual paramter) == Param(nominal_parameter_value)
        
    def nmpc_sIpopt_update(self, src='estimated'): 
        if src == 'estimated':
            perturbed_state = self.curr_estate
        else:
            perturbed_state = self.curr_rstate
        for x in self.states:
            i = 1
            x0 = getattr(self.olnmpc,x)
            x_icc = getattr(self.olnmpc,x+'_icc')
            for j in self.state_vars[x]:           
                self.olnmpc.sens_state_0.set_value(x0[(1,0)+j], i)
                self.olnmpc.sens_state_1.set_value(x0[(1,0)+j], i)
                self.olnmpc.sens_state_value_1.set_value(x0[(1,0)+j], perturbed_state[(x,j)])
                if j == ():
                    self.olnmpc.sens_init_constr.set_value(x_icc, i)
                else:
                    self.olnmpc.sens_init_constr.set_value(x_icc[j], i)    
                i += 1
                 
        before_dict = {}
        for u in self.u:
            control = getattr(self.olnmpc,u)
            before_dict[u] = control[1].value
            
        sIP = SolverFactory('ipopt_sens', solver_io = 'nl')
        sIP.options['run_sens'] = 'yes'
        sIP.options['tol'] = 1e-8
        sIP.options["halt_on_ampl_error"] = "yes"
        sIP.options["print_user_options"] = "yes"
        sIP.options["linear_solver"] = "ma57"
        results = sIP.solve(self.olnmpc, tee = True)
        self.olnmpc.solutions.load_from(results)
        
        self.nmpc_trajectory[self.iterations,'sIpopt'] = [str(results.solver.status),str(results.solver.termination_condition)]
        #flho: augmented
        
        after_dict = {}
        difference_dict = {}
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            after_dict[u] = control[1].value
            difference_dict[u] = after_dict[u] - before_dict[u]
        
        # save updated controls to current_control_info
        # apply clamping strategy if controls violate physical bounds
        for u in self.u:
            control = getattr(self.olnmpc, u)
            control[1].value = self.olnmpc.sens_sol_state_1.get(control[1])
            if control[1].value > control[1].ub:
                if control[1].ub == None: # number > None == True 
                    pass
                else:
                    control[1] = control[1].ub
            elif control[1].value < control[1].lb: #  number < None == False
                control[1].value = control[1].lb
            else:
                pass

        applied = {}
        for u in self.u:
            control = getattr(self.olnmpc,u)
            applied[u] = control[1].value
            self.curr_u[u] = control[1].value
            
        return before_dict, after_dict, difference_dict, applied            

            
    def sens_dot_nmpc(self):
        self.journalizer("I", self._c_it, "sens_dot_nmpc", "Set-up")
        if hasattr(self.olnmpc, "npdp"):
            self.olnmpc.npdp.clear()
        else:
            self.olnmpc.npdp = Suffix(direction=Suffix.EXPORT)

        for x in self.states:
            con_name = x + "_icc"
            con_ = getattr(self.olnmpc, con_name)
            for j in self.state_vars[x]:
                if j == ():
                    con_.set_suffix_value(self.olnmpc.npdp, self.curr_state_offset[(x, j)])
                else:
                    con_[j].set_suffix_value(self.olnmpc.npdp, self.curr_state_offset[(x, j)])

        if hasattr(self.olnmpc, "f_timestamp"):
            self.olnmpc.f_timestamp.clear()
        else:
            self.olnmpc.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        self.olnmpc.set_suffix_value(self.olnmpc.f_timestamp, self.int_file_nmpc_suf)

        self.olnmpc.f_timestamp.display(ostream=sys.stderr)

        self.journalizer("I", self._c_it, "sens_dot_nmpc", self.olnmpc.name)

        before_dict = {}
        for u in self.u:
            control = getattr(self.olnmpc,u)
            before_dict[u] = control[1].value

        results = self.dot_driver.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)
        
        after_dict = {}
        difference_dict = {}
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            after_dict[u] = control[1].value
            difference_dict[u] = after_dict[u] - before_dict[u]
                        
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
    
        applied = {}
        for u in self.u:
            control = getattr(self.olnmpc,u)
            applied[u] = control[1].value
            self.curr_u[u] = control[1].value
            
        return before_dict, after_dict, difference_dict, applied
        
            
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
        
        # TROUBLESHOOTING
        #self.olnmpc.s_mw.pprint()
        #self.olnmpc.s_PO_ptg.pprint()
        #self.olnmpc.s_unsat.pprint()
        #self.olnmpc.eps.pprint()
        
    def stall_strategy(self, strategy, cmv=1e-04, **kwargs):  # Fix the damn stall strategy
        """Suggested three strategies: Change weights, change matrices, change linear algebra"""
        self._stall_iter += 1
        self.journalizer("I", self._c_it, "stall_strategy", "Solver Stalled. " + str(self._stall_iter) + " Times")
        if strategy == "increase_weights":
            spf = 0
            ma57_as = "no"
            sw = self.olnmpc.s_w
            cw = self.olnmpc.c_w
            sw.value += sw.value
            cw.value += cw.value
            if sw.value > 1e06 or cw.value > 1e06:
                return 1
        elif strategy == "recompute_matrices":
            cmv += 1e04 * 5
            self.load_qk(max_qval=cmv)
        elif strategy == "linear_algebra":
            spf = 1
            ma57_as = "yes"

        retval = self.solve_d(self.olnmpc, max_cpu_time=300,
                              small_pivot_flag=spf,
                              ma57_automatic_scaling=ma57_as,
                              want_stime=True,
                              rep_timing=True)
        if retval == 0:
            return 0
        else:
            if self._stall_iter > 10:
                self.journalizer("I", self._c_it, "stall_strategy",
                                 "Max number of tries reached")
                sys.exit()
            self.stall_strategy("increase_weights")
            

    def find_target_ss(self, ref_state=None, **kwargs):
        """Attempt to find a second steady state
        Args:
            ref_state (dict): Contains the reference state with value key "state", (j,): value
            kwargs (dict): Optional arguments
        Returns
            None"""

        if ref_state:
            self.ref_state = ref_state
        else:
            if not ref_state:
                self.journalizer("W", self._c_it, "find_target_ss", "No reference state was given, using default")
            if not self.ref_state:
                self.journalizer("W", self._c_it, "find_target_ss", "No default reference state was given, exit")
                sys.exit()

        weights = dict.fromkeys(self.ref_state.keys())
        for i in self.ref_state.keys():
            v = getattr(self.ss, i[0])
            vkey = i[1]
            vss0 = value(v[(1, 1) + vkey])
            val = abs(self.ref_state[i] - vss0)
            if val < 1e-09:
                val = 1e+06
            else:
                val = 1/val
            weights[i] = val

        if bool(kwargs):
            pass
        else:
            self.journalizer("W", self._c_it, "find_target_ss", "Default-weights are being used")

        weights = kwargs.pop("weights", weights)

        self.journalizer("I", self._c_it, "find_target_ss", "Attempting to find steady state")

        del self.ss2
        self.ss2 = self.d_mod(1, 1, steady=True)
        self.ss2.name = "ss2 (reference)"
        for u in self.u:
            cv = getattr(self.ss2, u)  #: Get the param
            c_val = [value(cv[i]) for i in cv.keys()]  #: Current value
            self.ss2.del_component(cv)  #: Delete the param
            self.ss2.add_component(u, Var(self.ss2.fe_t, initialize=lambda m, i: c_val[i-1]))
            self.ss2.equalize_u(direction="r_to_u")
            cc = getattr(self.ss2, u + "_c")  #: Get the constraint
            ce = getattr(self.ss2, u + "_e")  #: Get the expression
            cv = getattr(self.ss2, u)  #: Get the new variable
            for k in cv.keys():
                cv[k].setlb(self.u_bounds[u][0])
                cv[k].setub(self.u_bounds[u][1])
            cc.clear()
            cc.rule = lambda m, i: cv[i] == ce[i]
            cc.reconstruct()

        self.ss2.create_bounds()
        self.ss2.equalize_u(direction="r_to_u")

        for vs in self.ss.component_objects(Var, active=True):  #: Load_guess
            vt = getattr(self.ss2, vs.getname())
            for ks in vs.keys():
                vt[ks].set_value(value(vs[ks]))
        ofexp = 0
        for i in self.ref_state.keys():
            v = getattr(self.ss2, i[0])
            val = value((v[(1, 1) + vkey]))
            vkey = i[1]
            ofexp += weights[i] * (v[(1, 1) + vkey] - self.ref_state[i])**2
            # ofexp += -weights[i] * (v[(1, 1) + vkey])**2 #- self.ref_state[i])**2
        self.ss2.obfun_ss2 = Objective(expr=ofexp, sense=minimize)

        tst = self.solve_d(self.ss2, iter_max=900, stop_if_nopt=False, halt_on_ampl_error=False)
        if tst != 0:
            self.ss2.display(filename="failed_ss2.txt")
            self.ss2.write(filename="failed_ss2.nl",
                           format=ProblemFormat.nl,
                           io_options={"symbolic_solver_labels": True})
            # sys.exit(-1)
        self.journalizer("I", self._c_it, "find_target_ss", "Target: solve done")
        for i in self.ref_state.keys():
            v = getattr(self.ss2, i[0])
            vkey = i[1]
            val = value(v[(1, 1) + vkey])
            print("target {:}".format(i[0]), "key {:}".format(i[1]), "weight {:f}".format(weights[i]),
                  "value {:f}".format(val))
        for u in self.u:
            v = getattr(self.ss2, u)
            val = value(v[1])
            print("target {:}".format(u), " value {:f}".format(val))
        self.update_targets_nmpc()

    def update_targets_nmpc(self):
        """Use the reference model to update  the current state and control targets"""
        for x in self.states:
            xvar = getattr(self.ss2, x)
            for j in self.state_vars[x]:
                self.curr_state_target[(x, j)] = value(xvar[1, 1, j])
        for u in self.u:
            uvar = getattr(self.ss2, u)
            self.curr_u_target[u] = value(uvar[1])

    def change_setpoint(self, ref_state, **kwargs):
        """Change the update the ref_state dictionary, and attempt to find a new reference state"""
        if ref_state:
            self.ref_state = ref_state
        else:
            if not ref_state:
                self.journalizer("W", self._c_it, "change_setpoint", "No reference state was given, using default")
            if not self.ref_state:
                self.journalizer("W", self._c_it, "change_setpoint", "No default reference state was given, exit")
                sys.exit()

        weights = dict.fromkeys(self.ref_state.keys())
        for i in self.ref_state.keys():
            v = getattr(self.ss, i[0])
            vkey = i[1]
            vss0 = value(v[(1, 1) + vkey])
            val = abs(self.ref_state[i] - vss0)
            if val < 1e-09:
                val = 1e+06
            else:
                val = 1/val
            weights[i] = val

        if bool(kwargs):
            pass
        else:
            self.journalizer("W", self._c_it, "find_target_ss", "Default-weights are being used")

        weights = kwargs.pop("weights", weights)

        ofexp = 0.0
        for i in self.ref_state.keys():
            v = getattr(self.ss2, i[0])
            vkey = i[1]
            ofexp += weights[i] * (v[(1, 1) + vkey] - self.ref_state[i]) ** 2

        self.ss2.obfun_ss2.set_value(ofexp)
        self.solve_d(self.ss2, iter_max=500, stop_if_nopt=True)

        for i in self.ref_state.keys():
            v = getattr(self.ss2, i[0])
            vkey = i[1]
            val = value(v[(1, 1) + vkey])
            print("target {:}".format(i[0]), "key {:}".format(i[1]), "weight {:f}".format(weights[i]),
                  "value {:f}".format(val))
        self.update_targets_nmpc()

    def compute_offset_state(self, src_kind="estimated"):
        """Missing noisy"""
        if src_kind == "estimated":
            for x in self.states:
                for j in self.state_vars[x]:
                    self.curr_state_offset[(x, j)] = self.curr_pstate[(x, j)] - self.curr_estate[(x, j)]
        elif src_kind == "real":
            for x in self.states:
                for j in self.state_vars[x]:
                    self.curr_state_offset[(x, j)] = self.curr_pstate[(x, j)] - self.curr_rstate[(x, j)]
        
        # can be removed if I am done with troubleshooting
        self.nmpc_trajectory[self.iterations,'state_offset'] = self.curr_state_offset
        dummy1, dummy2, dummy3 = deepcopy(self.curr_state_offset), deepcopy(self.curr_pstate), deepcopy(self.curr_estate)
        return  dummy1, dummy2, dummy3
    
    def print_r_nmpc(self):
        self.journalizer("I", self._c_it, "print_r_nmpc", "Results at" + os.getcwd())
        self.journalizer("I", self._c_it, "print_r_nmpc", "Results suffix " + self.res_file_suf)
        # print(self.soi_dict)
        for k in self.ref_state.keys():
            self.soi_dict[k].append(self.curr_soi[k])
            self.sp_dict[k].append(self.curr_sp[k])

        # for u in self.u:
        #     self.u_dict[u].append(self.curr_u[u])
        #     print(self.curr_u[u])

        with open("res_nmpc_rs_" + self.res_file_suf + ".txt", "a") as f:
            for k in self.ref_state.keys():
                i = self.soi_dict[k]
                iv = str(i[-1])
                f.write(iv)
                f.write('\t')
            for k in self.ref_state.keys():
                i = self.sp_dict[k]
                iv = str(i[-1])
                f.write(iv)
                f.write('\t')
            for u in self.u:
                i = self.curr_u[u]
                iv = str(i)
                f.write(iv)
                f.write('\t')
            for u in self.u:
                i = self.curr_ur[u]
                iv = str(i)
                f.write(iv)
                f.write('\t')
            f.write('\n')
            f.close()

        with open("res_nmpc_offs_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.states:
                for j in self.state_vars[x]:
                    i = self.curr_state_offset[(x, j)]
                    iv = str(i)
                    f.write(iv)
                    f.write('\t')
            f.write('\n')
            f.close()
        # with open("res_nmpc_u_" + self.res_file_suf + ".txt", "a") as f:
        #     for u in self.u:
        #         for i in range(0, len(self.u_dict[u])):
        #             iv = str(self.u_dict[u][i])
        #             f.write(iv)
        #             f.write('\t')
        #         f.write('\n')
        #     f.close()

    def update_soi_sp_nmpc(self):
        """States-of-interest and set-point update"""
        if bool(self.soi_dict):
            pass
        else:
            for k in self.ref_state.keys():
                self.soi_dict[k] = []

        if bool(self.sp_dict):
            pass
        else:
            for k in self.ref_state.keys():
                self.sp_dict[k] = []

        for k in self.ref_state.keys():
            vname = k[0]
            vkey = k[1]
            var = getattr(self.d1, vname)
            #: Assuming the variable is indexed by time
            self.curr_soi[k] = value(var[(1, self.ncp_t) + vkey])
        for k in self.ref_state.keys():
            vname = k[0]
            vkey = k[1]
            var = getattr(self.ss2, vname)
            #: Assuming the variable is indexed by time
            self.curr_sp[k] = value(var[(1, 1) + vkey])
        self.journalizer("I", self._c_it, "update_soi_sp_nmpc", "Current offsets:")
        for k in self.ref_state.keys():
            #: Assuming the variable is indexed by time
            self.curr_off_soi[k] = 100 * abs(self.curr_soi[k] - self.curr_sp[k])/abs(self.curr_sp[k])
            print("\tCurrent offset \% \% \t", k, self.curr_off_soi[k])

        for u in self.u:
            ur = getattr(self.ss2, u)
            self.curr_ur[u] = value(ur[1])

    def method_for_nmpc_simulation(self):
        pass
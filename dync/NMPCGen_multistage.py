#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:54:18 2017

@author: flemmingholtorf
"""
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
        
        # regularization
        self.delta_u = False
        
        #timestep bounds
        self.tf_bounds = kwargs.pop('tf_bounds',(10.0,30.0))
        
        # multistage
        dummy_tree = {}
        for i in range(1,self.nfe_t+1):
            dummy_tree[i,1] = (i-1,1,1,{'p':1.0,'i':1.0})
        self.st = kwargs.pop('scenario_tree', dummy_tree)
        self.multistage = kwargs.pop('multistage', True)
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
        self.reference_control_trajectory = {} # {('control',(fe,j)):value)}
        self.storage_model = object()
        self.initial_values = {}
        self.nominal_parameter_values = {}
        
        # for advanced step: 
        self.forward_simulation_model = self.d_mod(1,self.ncp_t, _t=self._t) # only need 1 element for forward simulation
        self.forward_simulation_model.name = 'forward simulation model'
        self.forward_simulation_model.create_output_relations()
        self.forward_simulation_model.create_bounds()
        self.forward_simulation_model.deactivate_pc()
        self.forward_simulation_model.deactivate_epc()
        self.forward_simulation_model.eobj.deactivate()
        self.forward_simulation_model.del_pc_bounds()
        self.simulation_trajectory = {}
        # organized as follows   
        #- for states:   {i, ('name',(index)): value} # actual state at time step i
        #- for controls: {i, 'name': value} 
                # --> control input valid between iteration i-1 and i 
                #     (i=1 --> is the first piecewise constant control input) 
        
        
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
        #- for states:   {i, ('name',(index)): value} # actual state at time step i
        #- for controls: {i, 'name': value} 
                # --> control input valid between iteration i-1 and i 
                #     (i=1 --> is the first piecewise constant control input) 
                
        self.pc_trajectory = {} 
        # organized as follows 
        #  {(pc,(iteration,collocation_point):value}
        
        self.iterations = 1
        self.nfe_mhe = 1
        self.nfe_t_0 = 0 # gets set during the recipe optimization

    def open_loop_simulation(self, sample_size = 10, **kwargs):
        # simulates the current
        # options: process_noise --> gaussian noise added to all states
        #          parameter noise --> noise added to the uncertain model parameters
        #                               or different ones
        #          input noise --> noise added to the inputs/controls
        # not exhaustive list, can be adapted/extended as one wishes
        # combination of all the above with noise in the initial point supporte
        initial_disturbance = kwargs.pop('initial_disturbance', {(x,j):0.0 for x in self.states for j in self.state_vars[x]})
        parameter_disturbance = kwargs.pop('parameter_disturbance', {})
        input_disturbance = kwargs.pop('input_disturbance',{})

        # deactivate constraints
        self.simulation_model = self.d_mod(self.nfe_t, self.ncp_t)
        self.simulation_model.deactivate_epc()
        self.simulation_model.deactivate_pc()
        self.simulation_model.eobj.deactivate()
        self.simulation_model.del_pc_bounds()
        self.simulation_model.clear_aux_bounds()
        self.simulation_model.fix_element_size.deactivate()
        
        # load initial guess from recipe optimization
        for var in self.simulation_model.component_objects(Var):
            var_ref = getattr(self.recipe_optimization_model, var.name)
            for key in var.index_set():
                var[key].value = var_ref[key].value
        
        nominal_initial_point = {}
        for x in self.states: 
            xic = getattr(self.recipe_optimization_model, x+'_ic')
            for j in self.state_vars[x]:
                key = j[1:] if j != (1,) else None
                nominal_initial_point[(x,j)] = xic[key].value
                    
        if parameter_disturbance != {}:
            for p in parameter_disturbance:
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
            print('#'*20)
            print(' '*5 + 'iter: ' + str(k))
            print('#'*20)
            pc_trajectory[k] = {}
            if initial_disturbance != {}:
                for x in self.states:
                    xic = getattr(self.simulation_model,x+'_ic')
                    for j in self.state_vars[x]:
                        key = j[1:] if j != (1,) else None
                        xic[key].value = nominal_initial_point[(x,j)] * (1 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j)]))
            if input_disturbance != {}:
                for u in input_disturbance:
                    control = getattr(self.simulation_model, u)
                    for i in range(1,self.nfe_t+1):
                        disturbance_noise = np.random.normal(loc=0.0, scale=input_disturbance[u])
                        control[i,1].value = self.reference_control_trajectory[u,(i,1)]*(1+disturbance_noise)
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
                           
                        
            
            self.simulation_model.tf.fix()
            self.simulation_model.equalize_u(direction="u_to_r")
            # run the simulation
            ip = SolverFactory("asl:ipopt")
            ip.options["halt_on_ampl_error"] = "yes"
            ip.options["print_user_options"] = "yes"
            ip.options["linear_solver"] = "ma57"
            ip.options["tol"] = 1e-8
            ip.options["max_iter"] = 3000
                
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
                self.simulation_model.troubleshooting()
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
        initial_disturbance = kwargs.pop('initial_disturbance', {(x,j):0.0 for x in self.states for j in self.x_vars[x]})
        parameter_disturbance = kwargs.pop('parameter_disturbance', {})
        state_disturbance = kwargs.pop('state_disturbance', {})
        input_disturbance = kwargs.pop('input_disturbance',{})
        
        #  generate the disturbance lording to specified scenario
        state_noise = {}
        input_noise = {}
        if disturbance_src == 'process_noise':
            if state_disturbance != {}: 
                for key in state_disturbances:
                    state_noise[key] = np.random.normal(loc=0.0, scale=state_disturbance[key])
                    # potentially implement truncation at 2 sigma
            else:
                for x in self.states: 
                    for j in self.x_vars[x]:
                        state_noise[(x,j)] = 0.0   
            
            for u in self.u:
                input_noise[u] = 0.0     
                
        elif disturbance_src == 'parameter_noise':
            for p in parameter_disturbance:
                disturbed_parameter = getattr(self.plant_simulation_model, p[0])               
                if first_call:
                    if p[1] != ():
                        self.nominal_parameter_values[p] = disturbed_parameter[p[1]].value
                    else:
                        self.nominal_parameter_values[p] = disturbed_parameter.value
                
                if (self.iterations-1)%parameter_disturbance[p][1] == 0:
                    sigma = parameter_disturbance[p][0]
                    rand = np.random.normal(loc=0.0, scale=sigma)
                    #truncation at 2 sigma
                    if abs(rand) > 2*sigma:
                        if rand < 0.0:
                            rand = -2.0 * sigma
                        else:
                            rand = 2.0 * sigma
                    if p[1] != ():
                        disturbed_parameter[p[1]].value = self.nominal_parameter_values[p] * (1 + rand)                 
                    else:
                        disturbed_parameter.value = self.nominal_parameter_values[p] * (1 + rand)
            for x in self.states:
                for j in self.x_vars[x]:
                    state_noise[(x,j)] = 0.0
                    
            for u in self.u:
                input_noise[u] = 0.0
        elif disturbance_src == 'input_disturbance': # input noise only
            for x in self.states:
                for j in self.x_vars[x]:
                    state_noise[(x,j)] = 0.0
            for u in self.u:
                input_noise[u] = np.random.normal(loc=0.0, scale=input_disturbance[u])
        else:
            print('NO DISTURBANCE SCENARIO SPECIFIED, NO NOISE ADDED ANYWHERE')                     
            for x in self.states:
                for j in self.x_vars[x]:
                        state_noise[(x,j)] = 0.0
            for u in self.u:
                input_noise[u] = 0.0
                    
        
        if first_call:
            for x in self.states:
                xic = getattr(self.plant_simulation_model,x+'_ic')
                xvar = getattr(self.plant_simulation_model,x)
                for j in self.state_vars[x]:
                    j_ic = j[:-1]
                    if j_ic == ():
                        xic.value = xic.value * (1.0 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j_ic)]))
                        aux = xic.value
                    else:
                        xic[j_ic].value = xic[j_ic].value * (1.0 + np.random.normal(loc=0.0, scale=initial_disturbance[(x,j_ic)]))
                        aux = xic[j_ic].value
                    # just for initialization purposes for the simulation
                    for k in range(0,self.ncp_t+1):
                        xvar[(1,k)+j].value = aux
            
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
                    elif type(key) == tuple:
                        try:
                            if len(key) > 2:
                                var[key] = var[(1,self.ncp_t)+key[2:]].value 
                            else:
                                var[key] = var[(1,key[1])].value
                        except KeyError:
                            continue
            
            # initialization of state trajectories
            # adding state noise if specified            
            for x in self.states:
                xic = getattr(self.plant_simulation_model,x+'_ic')
                x_var = getattr(self.plant_simulation_model,x)
                for j in self.x_vars[x]:
                    if j == (): 
                        xic.value = x_var[(1,self.ncp_t)+j+(1,)].value * (1.0 + state_noise[(x,j)])#+ noise # again tailored to RADAU nodes
                    else:
                        xic[j].value = x_var[(1,self.ncp_t)+j+(1,)].value * (1.0 + state_noise[(x,j)])# + noise # again tailored to RADAU nodes
                    # for initialization: take constant values + leave time invariant values as is!
                    for k in range(0,self.ncp_t+1):
                        x_var[(1,k)+j+(1,)].value = x_var[(1,self.ncp_t)+j+(1,)].value
           
        # result is the previous olnmpc solution 
        #    --> therefore provides information about the sampling interval
        # 1 element model for forward simulation
        self.plant_simulation_model.tf[1,1].fix(result['tf', (1,1)])

        # FIX(!!) the controls, path constraints are deactivated
        for u in self.u:
            control = getattr(self.plant_simulation_model,u)
            control[1,1].fix(result[u,(1,1)]*(1.0+input_noise[u]))
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
        
        #
        if [str(out.solver.status), str(out.solver.termination_condition)] != ['ok','optimal']:
            self.plant_simulation_model.clear_all_bounds()
            out = ip.solve(self.plant_simulation_model, tee = True)
        
        self.plant_trajectory[self.iterations,'solstat'] = [str(out.solver.status), str(out.solver.termination_condition)]
        self.plant_trajectory[self.iterations,'tf'] = self.plant_simulation_model.tf[1,1].value
        
        # safe results of the plant_trajectory dictionary {number of iteration, (x,j): value}
        for u in self.u:
            control = getattr(self.plant_simulation_model,u)
            self.plant_trajectory[self.iterations,u] = control[1,1].value #(control valid between 0 and tf)
        for x in self.states:
            xvar = getattr(self.plant_simulation_model, x)
            for j in self.state_vars[x]:
                    self.plant_trajectory[self.iterations,(x,j[:-1])] = xvar[(1,3)+j].value 
                    # setting the current real value w/o measurement noise
                    self.curr_rstate[(x,j)] = xvar[(1,3)+j].value 
                    
        # to monitor path constraints if supplied:
        if self.path_constraints != []:
            for pc_name in self.path_constraints:
                pc_var = getattr(self.plant_simulation_model, pc_name)
                for index in pc_var.index_set():
                    self.pc_trajectory[(pc_name,(self.iterations,index[1:-1]))] = pc_var[index].value
            
            for cp in range(self.ncp_t+1):
                self.pc_trajectory[('tf',(self.iterations,cp))] = self.plant_simulation_model.tau_i_t[cp]*self.plant_simulation_model.tf[1,1].value
            
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
        for x in self.states:
            xic = getattr(self.forward_simulation_model,x+'_ic')
            xvar = getattr(self.forward_simulation_model,x)
            for j in self.x_vars[x]:
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
            self.forward_simulation_model.clear_all_bounds()
            out = ip.solve(self.forward_simulation_model, tee = True, symbolic_solver_labels=True)

        self.simulation_trajectory[self.iterations,'solstat'] = [str(out.solver.status), str(out.solver.termination_condition)]
        
        # save the simulated state as current predicted state
        # implicit assumption of RADAU nodes
        for x in self.states:
            xvar = getattr(self.forward_simulation_model, x)
            for j in self.state_vars[x]:
                    self.curr_pstate[(x,j)] = xvar[(1,self.ncp_t)+j].value  # implicit assumption of RADAU nodes
         
    def create_tf_bounds(self,m):
        for index in m.tf.index_set():
            m.tf[index].setlb(self.tf_bounds[0])
            m.tf[index].setub(self.tf_bounds[1])
            
    def recipe_optimization(self, multimodel=False):
        self.nfe_t_0 = self.nfe_t # set self.nfe_0 to keep track of the length of the reference trajectory
        self.generate_state_index_dictionary()
        self.recipe_optimization_model = self.d_mod(self.nfe_t,self.ncp_t,scenario_tree = self.st, s_max = self.s_used, nr = self.nr)#self.d_mod(self.nfe_t, self.ncp_t, scenario_tree = self.st)
        self.recipe_optimization_model.initialize_element_by_element()
        self.recipe_optimization_model.create_bounds()
        self.recipe_optimization_model.clear_aux_bounds()
        self.recipe_optimization_model.create_output_relations()
        self.create_tf_bounds(self.recipe_optimization_model)
        
        if multimodel:
            self.recipe_optimization_model.multimodel()
        
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
        for x in self.states:
            xvar = getattr(self.recipe_optimization_model,x)
            for j in self.state_vars[x]:
                self.curr_pstate[(x,j)] = xvar[(1,self.ncp_t)+j].value
                self.initial_values[(x,j)] = xvar[(1,self.ncp_t)+j].value
        
        
    def get_tf(self,s):
        t = [] 
        for i in self.recipe_optimization_model.fe_t:
            if i == 1:
                t.append(self.recipe_optimization_model.tf[i,s].value)
            else:
                t.append(t[i-1-1] + self.recipe_optimization_model.tf[i,s].value)
        return t
            
    
    def set_reference_state_trajectory(self,input_trajectory):
        # input_trajectory = {('state',(fe,cp,j)):value} ---> this format
        self.reference_state_trajectory = input_trajectory

    def get_state_trajectory(self,d_mod):
        output = {}
        for state_name in self.states:
            xvar = getattr(d_mod,state_name)
            for key in xvar.index_set():
                if key[1] == self.ncp_t:
                    try:
                        output[(state_name,key)] = xvar[key].value
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
            u = getattr(d_mod,control_name)
            for key in u.index_set():
                try:
                    output[(control_name,key)] = u[key].value
                except KeyError:
                    print('something went wrong during get_state_trajectory')
                    continue
        return output           
     
    def generate_state_index_dictionary(self):
        # generates a dictionary = {'name':[indices except fe,cp] if only 1 add. index (j,) if none ()}
        for x in self.states:
            self.state_vars[x] = []
            try:
                xv = getattr(self.forward_simulation_model, x)
            except AttributeError:  # delete this
                continue
            for j in xv.keys():
                if j[1] == 0:
                    if type(j[2:]) == tuple:
                        self.state_vars[x].append(j[2:])
                    else:
                        self.state_vars[x].append((j[2:],))
                else:
                    continue
         
    def create_enmpc(self):
        self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t, scenario_tree = self.st, s_max = self.s_used, nr = self.nr)
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
                        expression += self.olnmpc.K_w_nmpc[1]*(self.nmpc_trajectory[self.iterations,u] - control[1,1])**2.0
              
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
        for x in self.states:
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
                if key[0] == 1: # only relevant for the first run, afterwards repeating
                    self.umpc_key[(u,key[1])] = k
                    k += 1
        
        #k = 0
        #for t in range(1, self.nfe_t + 1):
        #    self.xmpc_l[t] = []
        #    for x in self.states:
        #        x_var = getattr(self.olnmpc, x)  #: State
                
                #for j in self.state_vars[x]:
                #    self.xmpc_l[t].append(x_var[(t, self.ncp_t) + j])
                #    if t == 1: # only relevant for the first run, afterwards repeating
                #        self.xmpc_key[(x, j)] = k
                #        k += 1
    
        # same for inputs
        #self.umpc_l = {}
        #self.umpc_key = {}
        #k = 0
        #for t in range(1, self.nfe_t + 1):
        #    self.umpc_l[t] = []
        #    for u in self.u:
        #        u_var = getattr(self.olnmpc, u)
        #        self.umpc_l[t].append(u_var[t])
        #        if t == 1: # only relevant for the first run, afterwards repeating
        #            self.umpc_key[u] = k
        #            k += 1
        
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
                        expression += self.olnmpc.K_w_nmpc[1]*(self.nmpc_trajectory[self.iterations,u] - control[1,1])**2.0
              
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

    def load_reference_trajectories(self):
        for x in self.states:
            xvar = getattr(self.olnmpc, x)
            for key in xvar.index_set():
                if key[1] != self.ncp_t:# implicitly assuming RADAU nodes
                    continue 
                if type(key[2:]) == tuple:
                    j = key[2:]
                else:
                    j = (key[2:],)
                fe = key[0]
                try:
                    self.olnmpc.xmpc_ref_nmpc[fe,self.xmpc_key[(x,j)]] = self.reference_state_trajectory[x,(fe+self.iterations,self.ncp_t)+j]
                    #self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0/(self.reference_state_trajectory[x,(fe+self.iterations,self.ncp_t)+j] + 0.01)**2
                    self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0
                except KeyError:
                    self.olnmpc.xmpc_ref_nmpc[fe,self.xmpc_key[(x,j)]] = self.reference_state_trajectory[x,(self.nfe_t_0,self.ncp_t)+j]
                    #self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0/(self.reference_state_trajectory[x,(self.nfe_t_0,self.ncp_t)+j] + 0.01)**2
                    self.olnmpc.Q_nmpc[self.xmpc_key[(x,j)]] = 1.0
                        
        
        for u in self.u:    
            uvar = getattr(self.olnmpc, u)
            for j in uvar.index_set():
                fe = j[0]
                try:
                    self.olnmpc.umpc_ref_nmpc[fe,self.umpc_key[u,j[-1]]] = self.reference_control_trajectory[u,(fe+self.iterations,j[-1])]
                    #self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(self.reference_control_trajectory[u,fe+self.iterations] + 0.01)**2
                    #self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(control[1].ub-control[1].lb)**2
                    self.olnmpc.R_nmpc[self.umpc_key[u,j[-1]]] = 1.0
                except KeyError:
                    self.olnmpc.umpc_ref_nmpc[fe,self.umpc_key[u,j[-1]]] = self.reference_control_trajectory[u,(self.nfe_t_0,j[-1])]
                    #self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(self.reference_control_trajectory[u,self.nfe_t_0] + 0.01)**2
                    #self.olnmpc.R_nmpc[self.umpc_key[u]] = 1.0/(control[1].ub-control[1].lb)**2
                    self.olnmpc.R_nmpc[self.umpc_key[u,j[-1]]] = 1.0
                    
    def set_regularization_weights(self, K_w = 1.0, Q_w = 1.0, R_w = 1.0):
        if self.obj_type == 'economic':
            for i in self.olnmpc.fe_t:
                self.olnmpc.K_w_nmpc[i] = K_w
        else:
            for i in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[i] = Q_w
                self.olnmpc.R_w_nmpc[i] = R_w
                self.olnmpc.K_w_nmpc[i] = K_w
                
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
    
    def cycle_ics(self):
        for x in self.states:
            xic = getattr(self.olnmpc, x+'_ic')
            for j in self.state_vars[x]:
                j_aux = j[:-1]
                if not(j_aux == ()):
                    xic[j_aux].value = self.curr_rstate[(x,j)]
                else:
                    xic.value = self.curr_rstate[(x,j)]

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
        
        # adapt parameters by on-line estimation
        if self.adapt_params and self.iterations > 1:
            for index in self.curr_epars:
                p = getattr(self.olnmpc, index[0])
                key = index[1] if index[1] != () else None
                p[key].value = self.curr_epars[index]
 
        # set initial value parameters in model olnmpc
        # set according to predicted state by forward simulation
        # a) values will not be overwritten in case advanced step is used
        # b) values will be overwritten by call of add_noise() in case advanced step is not used
        for x in self.states:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.state_vars[x]:
                j_ic = j[:-1]
                if not(j_ic == ()):
                    xic[j_ic].value = self.curr_pstate[(x,j)]
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
        #self.olnmpc.write_nl()         
        if not(str(results.solver.status) == 'ok' and str(results.solver.termination_condition) == 'optimal'):
            print('olnmpc failed to converge')
            
        # save relevant results
        # self.iterations + 1 holds always if solve_olnmpc called at the correct time
        self.nmpc_trajectory[self.iterations,'solstat'] = [str(results.solver.status),str(results.solver.termination_condition)]
        self.nmpc_trajectory[self.iterations+1,'tf'] = self.nmpc_trajectory[self.iterations,'tf'] + self.olnmpc.tf[1,1].value
        self.nmpc_trajectory[self.iterations,'eps'] = [self.olnmpc.eps[1,1].value,self.olnmpc.eps[2,1].value,self.olnmpc.eps[3,1].value]  
        if self.obj_type == 'economic':
            self.nmpc_trajectory[self.iterations,'obj_value'] = value(self.olnmpc.eobj)
        else:
            self.nmpc_trajectory[self.iterations,'obj_value'] = value(self.olnmpc.objfun_nmpc)
  
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.nmpc_trajectory[self.iterations+1,u] = control[1,1].value # control input between timestep i-1 and i
        
        # initial state is saved to keep track of how good the estimates are
        # here only self.iteration since its the beginning of the olnmpc run
        for x in self.states:
            xic = getattr(self.olnmpc, x + '_ic')
            for j in self.state_vars[x]:
                j_ic = j[:-1]
                if j_ic == ():
                    self.nmpc_trajectory[self.iterations,(x,j_ic)] = xic.value   
                else:
                    self.nmpc_trajectory[self.iterations,(x,j_ic)] = xic[j_ic].value
                    
        # save the control result as current control input
        for u in self.u:
            control = getattr(self.olnmpc,u)
            self.curr_u[u] = control[1,1].value
        
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
            uv[1,1].set_suffix_value(m.dof_v, 1)

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

        # SAVE OLD
        before_dict = {}
        for u in self.u:
            control = getattr(self.olnmpc,u)
            before_dict[u] = control[1,1].value
            
        results = self.dot_driver.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)

        # SAVE NEW
        after_dict = {}
        difference_dict = {}
        
        for u in self.u:
            control = getattr(self.olnmpc,u)
            after_dict[u] = control[1,1].value
            difference_dict[u] = after_dict[u] - before_dict[u]


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
        
        if abs(self.olnmpc.u2[1,1].value - self.nmpc_trajectory[self.iterations+1,'u2']) > 1.0:
            print('solution way off')
            print(self.curr_state_offset)
            #sys.exit()
            
        applied = {}
        for u in self.u:
            control = getattr(self.olnmpc, u)
            self.curr_u[u] = control[1,1].value    
            applied[u] = control[1,1].value
            
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
        
    def create_nmpc_sIpopt_suffixes(self):
        m = self.olnmpc
        m.sens_state_0 = Suffix(direction=Suffix.EXPORT) # enumerates parameters that will be perturbed --> unique mapping number --> Var(actual parameter)
        m.sens_state_1 = Suffix(direction=Suffix.EXPORT) # enumerates parameters that are pertured --> mapping has to be equivalent to sens_state_0
        m.sens_state_value_1 = Suffix(direction=Suffix.EXPORT) # holds the perturbed parameter values, has to be set for the same variables as sens_state_1 is
        m.sens_sol_state_1  = Suffix(direction=Suffix.IMPORT) # holds the updated variables and lagrange multipliers for inequalities (and equalities (would be relatively meaninglessS?)) 
        m.sens_init_constr  = Suffix(direction=Suffix.EXPORT) # flag constraint that are artificial Var(actual paramter) == Param(nominal_parameter_value)
        
    def SBWCS_hyrec(self, pc = [], epc = [], **kwargs):
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
         
        bounds = kwargs.pop('par_bounds',{}) # uncertainty bounds
        crit = kwargs.pop('crit','overall')
        # prepare sensitivity computation
        self.olnmpc.eps_pc.fix()
        self.olnmpc.eps.fix()
        for u in self.u:
            u_var = getattr(self.olnmpc,u)
            u_var.fix()
        self.olnmpc.tf.fix()
        self.olnmpc.clear_all_bounds()
        
        # deactivate nonanticipativity
        for u in self.u:
            non_anti = getattr(self.olnmpc, 'non_anticipativity_' + u)
            non_anti.deactivate()
        self.olnmpc.fix_element_size.deactivate()
        self.olnmpc.non_anticipativity_tf.deactivate()
            
        # set suffixes
        self.olnmpc.var_order = Suffix(direction=Suffix.EXPORT)
        self.olnmpc.dcdp = Suffix(direction=Suffix.EXPORT)
        
        # tailored for two stage
        # include parameters at the first stage after the robust horizon     
        i = 0
        cols ={}
        for p in self.p_noisy:
            for key in self.p_noisy[p]:
                if key != ():
                    dummy = 'dummy_constraint_p_' + p + '_' + str(key[0])
                else:
                    dummy = 'dummy_constraint_p_' + p
                dummy_con = getattr(self.olnmpc, dummy)
                for index in dummy_con.index_set():
                    #index[0] = time_step \in {2, ... ,nfe+1}
                    #index[1] = scenario \in {1, ... , s_per_branch}
                    if index[0] > 1 and \
                       index[0] < self.nr + 2 and \
                       index[-1] == 1:
                        self.olnmpc.dcdp.set_value(dummy_con[index], i+1)
                        cols[i] = (p,key+index)
                        i += 1
           
        # column i in sensitivity matrix corresponds to paramter p
        cols_r = {value:key for key, value in cols.items()}
        tot_cols = i
        
        i = 0
        rows = {}
        #path constraints only at next stage
        for k in pc:
            s = getattr(self.olnmpc, 's_'+k)
            for index in s.index_set():
                if not(s[index].stale): # only take
                    #index[0] = time_step \in {2, ... ,nfe+1}
                    #index[1] = collocation point = all of them here  (alternatively ncp_t: consider only endpoint of interval)
                    #index[-1] = scenario: only nominal scenario as base for linearization
                    #index[2:-1] = additional indices = all there are
                    if index[0] > 1 and \
                       index[0] < self.nr + 2 and \
                       index[-1] == 1: #
                        self.olnmpc.var_order.set_value(s[index], i+1)
                        rows[i] = ('s_'+ k,index)
                        i += 1
                        
        # endpoint constraints only in last iteration
        # epc dont make sense in this framework
        #if self.iterations == self.nfe_t_0 - 1: 
#        for k in epc:
#            s = getattr(self.olnmpc, 's_'+k)
#            for index in s.index_set():
#                if not(s[index].stale): # only take
#                    if index[-1] == 1: # only nominal scenario as base for linearization
#                    self.olnmpc.var_order.set_value(s[index], i+1)
#                    rows[i] = ('s_'+ k,index)
#                    i += 1
                        
        # row j in sensitivity matrix corresponds to rhs of constraint x 
        rows_r = {value:key for key, value in rows.items()}
        tot_rows = i

        # compute sensitivity matrix (ds/dp , rows = s const., cols = p const.)
        k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
        k_aug.options["compute_dsdp"] = ""
        k_aug.solve(self.olnmpc, tee=True)
            
        # no idea if necessery, I am lost in the code
        self.olnmpc.eps.unfix()
        self.olnmpc.eps_pc.unfix()
        for u in self.u:
            u_var = getattr(self.olnmpc,u)
            u_var.unfix()
        self.olnmpc.tf.unfix()
        self.olnmpc.create_bounds()
        self.create_tf_bounds(self.olnmpc)
        self.olnmpc.clear_aux_bounds()
        
        # activate non_anticipativity
        for u in self.u:
            non_anti = getattr(self.olnmpc, 'non_anticipativity_' + u)
            non_anti.activate()
        self.olnmpc.fix_element_size.activate()    
        self.olnmpc.non_anticipativity_tf.activate()     
        
        # read sensitivity matrix
        sens = np.zeros((tot_rows,tot_cols))
        with open('dxdp_.dat') as f:
            reader = csv.reader(f, delimiter="\t")
            i = 1
            for row in reader:
                k = 1
                for col in row[1:]:
                    sens[i-1][k-1] = -float(col) #indices start at 0
                    k += 1                       #different sign since g(x) == -slack
                i += 1
        
        # compute worst-case scenarios 
        delta_p_wc = {}
        con_vio = {}
        for i in rows: # constraints
            con = rows[i] # ('s_name', index)
            c_stage = con[1][0]
            c_scen = con[1][-1]
            dsdp = sens[i][:].transpose() # gets row i in dsdp, transpose not required
            delta_p_wc_iter = {}
            vertex = {}
            aux = 0.0
            for j in cols: # parameters
                par = cols[j] # cols[j] = ('pname', index)
                p = par[0] 
                key = par[1][:-2]
                p_stage = par[1][-2]
                p_scen = par[1][-1]
                if [p_stage,p_scen] == [c_stage,c_scen]: # only compute for relevant parameters
                    p_var = getattr(self.olnmpc, 'p_' + p)
                    delta = p_var[par[1]].value - 1.0
                    # delta = 0.0 # change if necessary
                    # bounds[par][0]: lower bound on delta_p
                    # bounds[par][1]: upper bound on delta_p
                    # shift depending on which corner was looked at before that
                    delta_p_wc_iter[p,key] = bounds[(p,key)][0] - delta if dsdp[j] < 0.0 else bounds[(p,key)][1] - delta
                    vertex[p,key] = 'L' if dsdp[j] < 0.0 else 'U'
                    aux += dsdp[j]*delta_p_wc_iter[p,key]
                else:
                    continue
            if crit == 'overall':
                s = getattr(self.olnmpc, con[0])
                con_vio[con] = -s[con[1]].value + aux
                delta_p_wc[con] = vertex
            elif crit == 'con':
                if key in con_vio:
                    if con_vio[con] < -s[con[1]].value + aux:
                        delta_p_wc = vertex
                        con_vio[con] = -s[con[1]].value + aux
                else:
                    delta_p_wc[con] = vertex
                    con_vio[con] = -s[con[1]].value + aux
            else:
                sys.exit('Error: Wrong specification of worst case criteria')
        print('cols',cols) 
        print('\n')
        print('rows',rows)
        print('\n')
        print('con_vio',con_vio)
        print('\n')
        print('delta_p_wc',delta_p_wc)
        scenarios = {}
        s_branch = {}
        s_stage = {0:1}
        # overall wc 
        if crit == 'overall':
            for i in range(2,min(self.nr+2,self.nfe_t+1)):
                con_vio_copy = {con:con_vio[con] for con in con_vio if con[1][0]==i}
                print('copy',i, con_vio_copy)
                for s in range(2,int(np.round(self.s_max**(1.0/self.nr)))+1): # scenario 1 is reserved for the nomnal sscenario
                    wc_scenario = max(con_vio_copy,key=con_vio_copy.get)
                    scenarios[i-1,s] = delta_p_wc[wc_scenario]
                    # remove scenario from scenarios:
                    con_vio_copy = {key: value for key, value in con_vio_copy.items() if (delta_p_wc[key] != delta_p_wc[wc_scenario])}
                    if len(con_vio_copy) == 0:
                        break
                s_branch[i-1] = s 
                s_stage[i-1] = s*s_stage[i-2]
        elif crit == 'con':            
        # wc for every constraint
            for i in range(2,min(self.nr+2,self.nfe_t+1)):
                s=1
                con_vio_copy = {con:con_vio[con] for con in con_vio if con[1][0]==i}
                scenarios_copy = {}
                for con in con_vio_copy:
                    scenarios[i-1,s] = delta_p_wc[con] if not(delta_p_wc[con] in scenarios_copy.values()) else 1
                    if scenarios[s]:
                        continue
                    scenarios_copy[s] = delta_p_wc[con]
                    s+=1
                s_branch[i-1] = s 
                s_stage[i-1] = s*s_stage[i-2]
        else:
            sys.exit('Error: Wrong specification of worst case criteria')
        print('scenarios',scenarios)
        
        
        self.s_used = s_stage[min(self.nr,self.nfe_t-1)]# nfe_t-1 since it is called before cycle_nmpc
        self.nmpc_trajectory[self.iterations, 's_max'] = s**self.nr
    
        # update scenario tree
        for i in range(1,self.nfe_t-1+1):
            if i < self.nr + 1:
                for s in range(1,s_stage[i]+1):
                    if s%s_branch[i]==1:
                        self.st[(i,s)] = (i-1,int(np.ceil(s/float(s_branch[i]))),True,{(p,key) : 1.0 for p in self.p_noisy for key in self.p_noisy[p]})
                    else:
                        scenario_key = s%s_branch[i] if s%s_branch[i] != 0 else s_branch[i] 
                        self.st[(i,s)] = (i-1,int(np.ceil(s/float(s_branch[i]))),False,{(p,key) : 1.0 + bounds[p,key][0] if scenarios[i,scenario_key][(p,key)]=='L' else 1 + bounds[p,key][1] for p in self.p_noisy for key in self.p_noisy[p]})
            else:
                for s in range(1,self.s_used+1):
                    self.st[(i,s)] = (i-1,s,True,self.st[(i-1,s)][3])
    
    def st_adaption(self, set_type=None, cons = [], **kwargs):
        epsilon = kwargs.pop('epsilon',0.2)
        shape_matrix = kwargs.pop('shape_matrix',self._scaled_shape_matrix)
        shape_matrix_indices = kwargs.pop('shape_matrix_indices',self.PI_indices)
        bounds = kwargs.pop('par_bounds')
        if set_type == 'ellipsoid':  
            self.SBSG_hyell(cons=cons,
                            shape_matrix=shape_matrix,
                            shape_matrix_indices=shape_matrix_indices,
                            epsilon=epsilon)
        elif set_type == 'rectangle':
            self.SBSG_hyrec(cons=cons,
                            par_bounds=bounds)
        else:
            sys.exit('unknown uncertainty set type')
    
        
    def SBSG_hyrec(self,cons = [], **kwargs):    
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
        if self.iterations > 1:
            m = self.olnmpc
        else:
            m = self.recipe_optimization_model
            
        bounds = kwargs.pop('par_bounds',{}) # uncertainty bounds
        
        # prepare sensitivity computation
        m.eps_pc.fix()
        m.eps.fix()
        for u in self.u:
            u_var = getattr(m,u)
            u_var.fix()
        m.tf.fix()
        m.clear_all_bounds()
        
        # deactivate nonanticipativity
        for u in self.u:
            non_anti = getattr(m, 'non_anticipativity_' + u)
            non_anti.deactivate()
        m.fix_element_size.deactivate()
        m.non_anticipativity_tf.deactivate()
            
        
        # set suffixes
        m.var_order = Suffix(direction=Suffix.EXPORT)
        m.dcdp = Suffix(direction=Suffix.EXPORT)
        i = 0
        cols ={}
        for p in self.p_noisy:
            for key in self.p_noisy[p]:
                if key != ():
                    dummy = 'dummy_constraint_p_' + p + '_' + str(key[0])
                else:
                    dummy = 'dummy_constraint_p_' + p
                dummy_con = getattr(m, dummy)
                for index in dummy_con.index_set():
                    if type(index) == tuple:
                        if index[-1] == 1: # only take the nominal trajectory
                            m.dcdp.set_value(dummy_con[index], i+1)
                            cols[i] = (p,key)
                            i += 1
                    else:
                        if index == 1: # only take the nominal trajectory
                            m.dcdp.set_value(dummy_con[index], i+1)
                            cols[i] = (p,key)
                            i += 1
                            
        # column i in sensitivity matrix corresponds to paramter p
        cols_r = {value:key for key, value in cols.items()}
        tot_cols = i
        
        i = 0
        rows = {}
        for k in cons:
            s = getattr(m, 's_'+k)
            for index in s.index_set():
                if not(s[index].stale): # only take
                    if type(index) == tuple:
                        if index[-1] == 1: # only take the nominal trajectory
                            m.var_order.set_value(s[index], i+1)
                            rows[i] = ('s_'+ k,index)
                            i += 1
                    else:
                        if index == 1:
                            m.var_order.set_value(s[index], i+1)
                            rows[i] = ('s_'+ k,index)
                            i += 1  
        # row j in sensitivity matrix corresponds to rhs of constraint x 
        rows_r = {value:key for key, value in rows.items()}
        tot_rows = i
        
        # compute sensitivity matrix (ds/dp , rows = s const., cols = p const.)
        k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
        k_aug.options["compute_dsdp"] = ""
        k_aug.solve(m, tee=True)
        
        # no idea if necessery, I am lost in the code
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
        m.fix_element_size.activate()    
        m.non_anticipativity_tf.activate()

        # read sensitivity matrix
        sens = np.zeros((tot_rows,tot_cols))
        with open('dxdp_.dat') as f:
            reader = csv.reader(f, delimiter="\t")
            i = 1
            for row in reader:
                k = 1
                for col in row[1:]:
                    sens[i-1][k-1] = -float(col) #indices start at 0
                    k += 1                       #different sign since g(x) == -slack
                i += 1 
        
        # compute solution vertices and constraint violations
        delta_p_wc = {}
        con_vio = {}
        for i in rows: # constraints
            con = rows[i] # ('s_name', index)
            dsdp = sens[i][:].transpose()
            delta_p_wc[con] = {}
            aux = 0.0
            for j in cols: # parameters
                par = cols[j] #('p_name', key)
                # bounds[par][0]: lower bound on delta_p
                # bounds[par][1]: upper bound on delta_p
                delta_p_wc[con][par] = bounds[par][0] if dsdp[j] < 0.0 else bounds[par][1]
                aux += dsdp[j]*delta_p_wc[con][par]
            s = getattr(m, con[0])
            con_vio[con] = -s[con[1]].value + aux 
                     
        
        scenarios = {}
        con_vio_copy = deepcopy(con_vio)
        for s in range(2,int(np.round(self.s_max**(1.0/self.nr)))+1): # scenario 1 is reserved for the nomnal sscenario
            wc_scenario = max(con_vio_copy,key=con_vio_copy.get)
            scenarios[s] = delta_p_wc[wc_scenario]
            # remove scenario from scenarios:
            con_vio_copy = {key: value for key, value in con_vio_copy.items() if (delta_p_wc[key] != delta_p_wc[wc_scenario])}
            if len(con_vio_copy) == 0:
                break
        
        self.s_used = s**self.nr
        self.nmpc_trajectory[self.iterations, 's_max'] = s**self.nr
        s_per_branch = s
            
        # update scenario tree
        for i in range(1,self.nfe_t+1):
            if i < self.nr + 1:
                for s in range(1,s_per_branch**i+1):
                    if s%s_per_branch==1:
                        self.st[(i,s)] = (i-1,int(np.ceil(s/float(s_per_branch))),True,{(p,key) : 1.0 for p in self.p_noisy for key in self.p_noisy[p]})
                    else:
                        key = s%s_per_branch 
                        self.st[(i,s)] = (i-1,int(np.ceil(s/float(s_per_branch))),False,{(p,key) : 1.0 + scenarios[s%(s_per_branch+1)][(p,key)] for p in self.p_noisy for key in self.p_noisy[p]})
            else:
                for s in range(1,s_per_branch**self.nr+1):
                    self.st[(i,s)] = (i-1,s,True,self.st[(i-1,s)][3])
        #print(scenarios)
            
    def SBSG_hyell(self, m, cons = [], **kwargs):
        # for ellipsoidal sets:
        # solve for all constraints (all timepoints) the optimization problem:
        #           max ds/dp^T * delta_p
        #    s.t. 
        #           delta_p^T * A * delta_p <= 1
        #
        #
        # solution analytically derivable:
        #               delta_p^* = A^{-1} * ds/dp / sqrt(ds/dp^T A^{-1} ds/dp)
        #
        # note: - A \in \in \mathbb{S}^{n_p}_{++} describes ellipsoidal uncertainty set
        #       - natural choice: A = 1/chisquare * V_p^{-1} in case normally distributed parameters 
        #       - red_hessian^-1 approx. V_p which ultimately is required
        #            - scaled_red_hessian \in \mathbb{S}^{n_p}_{++} obtained from KKT matrix by k_aug (efficient, backsolves basically)
        #       - ds/dp \in \mathbb{R}^{n_p} obtained from k_aug

        # Iteratively populate scenarios:
            # include scenario that would result in maximum constraint violation in first order approximation:
            #           - approx. violation: sqrt(ds/dp^T A^{-1} ds/dp)
            #
            # exclude everything that lies inside a ball neighborhood around WC-scenario parameter realization
            #           
            # repeat until as many scenarios as possible are included 
        if self.iterations > 1:
            m = self.olnmpc
        else:
            m = self.recipe_optimization_model
        
        epsilon = kwargs.pop('epsilon',0.2)
        shape_matrix = kwargs.pop('shape_matrix',self._scaled_shape_matrix)
        shape_matrix_indices = kwargs.pop('shape_matrix_indices',self.PI_indices)
        # set shape matrix             
        # prepare sensitivity computation
        m.eps_pc.fix()
        m.eps.fix()
        for u in self.u:
            u_var = getattr(m,u)
            u_var.fix()
        m.tf.fix()
        m.clear_all_bounds()
        
        # deactivate nonanticipativity
        for u in self.u:
            non_anti = getattr(m, 'non_anticipativity_' + u)
            non_anti.deactivate()
        m.fix_element_size.deactivate()
        m.non_anticipativity_tf.deactivate()
            
        # set suffixes
        m.var_order = Suffix(direction=Suffix.EXPORT)
        m.dcdp = Suffix(direction=Suffix.EXPORT)
        i = 0
        cols ={}
        for p in self.p_noisy:
            for key in self.p_noisy[p]:
                if key != ():
                    dummy = 'dummy_constraint_p_' + p + '_' + str(key[0])
                else:
                    dummy = 'dummy_constraint_p_' + p
                dummy_con = getattr(m, dummy)
                for index in dummy_con.index_set():
                    if type(index) == tuple:
                        if index[-1] == 1: # only take the nominal trajectory
                            m.dcdp.set_value(dummy_con[index], i+1)
                            cols[i] = (p,key)
                            i += 1
                    else:
                        if index == 1:
                            m.dcdp.set_value(dummy_con[index], i+1)
                            cols[i] = (p,key)
                            i += 1
        cols_r = {value:key for key, value in cols.items()}
        tot_cols = i
        
        i = 0
        rows = {}
        for k in cons:
            s = getattr(m, 's_'+k)
            for index in s.index_set():
                if not(s[index].stale): # only take
                    if type(index) == tuple:
                        if index[-1] == 1: # only take the nominal trajectory
                            m.var_order.set_value(s[index], i+1)
                            rows[i] = ('s_'+ k,index)
                            i += 1
                    else:
                        if index == 1:
                            m.var_order.set_value(s[index], i+1)
                            rows[i] = ('s_'+ k,index)
                            i += 1  
        rows_r = {value:key for key, value in rows.items()}
        tot_rows = i
        
        # compute sensitivity matrix (ds/dp , rows = s const., cols = p const.)
        k_aug = SolverFactory("k_aug",executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
        k_aug.options["compute_dsdp"] = ""
        k_aug.solve(m, tee=True)
        
        # no idea if necessery, I am lost in the code
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
        m.fix_element_size.activate()    
        m.non_anticipativity_tf.activate()
        
        # read sensitivity matrix
        sens = np.zeros((tot_rows,tot_cols))
        with open('dxdp_.dat') as f:
            reader = csv.reader(f, delimiter="\t")
            i = 1
            for row in reader:
                k = 1
                for col in row[1:]:
                    sens[i-1][k-1] = -float(col) #indices start at 0
                    k += 1
                i += 1  
                
        # reorder self._PI (reduced hessian) and invert
        A = np.zeros((tot_cols,tot_cols))  # can be sped up by exploiting symmetry
        for index1 in cols:
            key_sens1 = cols[index1]
            key_PI1 = shape_matrix_indices[key_sens1]
            for index2 in cols:
                key_sens2 = cols[index2]
                key_PI2 = shape_matrix_indices[key_sens2]
                A[index1][index2] = shape_matrix[key_PI1][key_PI2] 
        
        # unnecessary when using the inverse of the reduced hessian directly, ask David about k_aug 
        A_inv = np.linalg.inv(A) # in principle not required, scaled inverse reduced hessian
        
        # compute worst case parameter realizations and expected violations:
        delta_p_wc = {}
        con_vio = {}
        for i in rows:
            con = rows[i] # ('s_name', index)
            # transpose is not required since python will adjust it according to the 
            # operation and matrix anyway
            # sens[i,:] draws the same thing as sens[i][:]
            # ATTENTION:
            # sens[i][:] sets the same thing as sens[:][i] (always row i)
            # if vectors are set use sens[i,:] sets all variables of a row
            # sens[:,i] sets all variables of a column
            dsdp = sens[i][:].transpose() # row i in dsdp, sensitivity of rhs to parameters
            aux = np.sqrt(np.dot(dsdp.transpose(),np.dot(A_inv,dsdp)))
            delta_p_wc[con] = np.dot(A_inv,dsdp)/aux
            # get current constraint violation (slack)
            s = getattr(m, con[0])
            # slack with different sign than rhs g(x) + s == 0.0
            con_vio[con] = -s[con[1]].value + aux
            
        # generate new set of worst case scenarios
        scenarios = {}
        con_vio_copy = deepcopy(con_vio)
        for s in range(2,int(np.round(self.s_max**(1.0/self.nr)))+1): # scenario 1 is reserved for the nomnal sscenario
            wc_scenario = max(con_vio_copy,key=con_vio_copy.get)
            scenarios[s] = delta_p_wc[wc_scenario]
            # What to do if con_vio_copy is empty?
            
            # a) remove all scenarios that are in epsilon-neighborhood of the just utilized scenario and decrease epsilon if set gets empty
            #while(True):
            #    con_vio_dummy = {key: value for key, value in con_vio_copy.items() if np.linalg.norm(delta_p_wc[key]-delta_p_wc[wc_scenario]) >= epsilon}
            #    if len(con_vio_dummy) < self.s_max + 1 - s:
            #        epsilon *= 0.8
            #    else:
            #        con_vio_copy = deepcopy(con_vio_dummy)
            #        break
            
            # b) drop scenarios if set gets empty
            con_vio_copy = {key: value for key, value in con_vio_copy.items() if np.linalg.norm(delta_p_wc[key]-delta_p_wc[wc_scenario]) >= epsilon}
            if len(con_vio_copy) == 0:
                break

        self.s_used = s**self.nr
        self.nmpc_trajectory[self.iterations, 's_max'] = s**self.nr
        s_per_branch = s
            
        # update scenario tree
        for i in range(1,self.nfe_t+1):
            if i < self.nr + 1:
                for s in range(1,s_per_branch**i+1):
                    if s%s_per_branch==1:
                        self.st[(i,s)] = (i-1,int(np.ceil(s/float(s_per_branch))),True,{(p,key) : 1.0 for p in self.p_noisy for key in self.p_noisy[p]})
                    else:
                        self.st[(i,s)] = (i-1,int(np.ceil(s/float(s_per_branch))),False,{(p,key) : 1.0 + scenarios[s%(s_per_branch+1)][(p,key)] for p in self.p_noisy for key in self.p_noisy[p]})
            else:
                for s in range(1,s_per_branch+1):
                    self.st[(i,s)] = (i-1,s,True,self.st[(i-1,s)][3])
            
            
        
    
    def nmpc_sIpopt_update(self, src='estimated'): 
        if src == 'estimated':
            perturbed_state = self.curr_estate
        else:
            perturbed_state = self.curr_rstate
        for x in self.states:
            i = 1
            x0 = getattr(self.olnmpc, x)
            x_icc = getattr(self.olnmpc, x+'_icc')
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
            before_dict[u] = control[1,1].value
            
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
            after_dict[u] = control[1,1].value
            difference_dict[u] = after_dict[u] - before_dict[u]
        
        # save updated controls to current_control_info
        # apply clamping strategy if controls violate physical bounds
        for u in self.u:
            control = getattr(self.olnmpc, u)
            control[1,1].value = self.olnmpc.sens_sol_state_1.get(control[1,1])
            if control[1,1].value > control[1,1].ub:
                if control[1,1].ub == None: # number > None == True 
                    pass
                else:
                    control[1,1] = control[1,1].ub
            elif control[1,1].value < control[1,1].lb: #  number < None == False
                control[1,1].value = control[1,1].lb
            else:
                pass

        applied = {}
        for u in self.u:
            control = getattr(self.olnmpc,u)
            applied[u] = control[1,1].value
            self.curr_u[u] = control[1,1].value

        return before_dict, after_dict, difference_dict, applied 
    

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
        
        # flho: modified
        self.nmpc_trajectory[self.iterations,'state_offset'] = self.curr_state_offset
        dummy1, dummy2, dummy3 = deepcopy(self.curr_state_offset), deepcopy(self.curr_pstate), deepcopy(self.curr_estate)
        return dummy1, dummy2, dummy3
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
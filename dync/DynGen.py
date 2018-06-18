#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:54:18 2017

@author: flemmingholtorf
"""
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, Set, Constraint, Expression, Param, Suffix, maximize
from pyomo.core.base import ConstraintList
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from pyomo.core.base import value
import numpy as np
import sys, time


__author__ = "@FHoltorf"

""" yet another day debugging """

class DynGen(object):
    def __init__(self, **kwargs):
        # multimodel option noch einbauen                
        self.d_mod = kwargs.pop('d_mod', None)

        self.multimodel = kwargs.pop('multimodel', False)

        self.nfe_t = self.nfe_t_0 = kwargs.pop('nfe_t', 24)
        self.ncp_t = kwargs.pop('ncp_t', 3) # usually use Radau IIA
        self.nfe_mhe = 1
        self._c_it = 1
        
        self.x = kwargs.pop('x', {})
        self.u = kwargs.pop('u', [])  #: The inputs (controls)
        
        self.tf_bounds = kwargs.pop('tf_bounds',(10.0,30.0))
        
        self.ipopt = SolverFactory("ipopt")
        self.asl_ipopt = SolverFactory("asl:ipopt")
   
        self.asl_ipopt.options["halt_on_ampl_error"] = "yes"
        self.asl_ipopt.options["print_user_options"] = "yes"
        self.asl_ipopt.options["linear_solver"] = "ma57"
        
        self.ipopt.options["print_user_options"] = "yes"
        
        self.ipopt_options = kwargs.pop("ipopt_options",{})
        for option in self.ipopt_options:
            self.asl_ipopt.options[option] = self.ipopt_options[option]
         
        with open("ipopt.opt", "w") as f:
            f.close()
            
        self.k_aug = SolverFactory("k_aug",
                                   executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
        self.k_aug_sens = SolverFactory("k_aug",
                                        executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/k_aug")
        self.dot_driver = SolverFactory("dot_driver",
                                        executable="/home/flemmingholtorf/KKT_matrix/k_aug/src/kmatrix/dot_driver/dot_driver")

        self.curr_u = dict.fromkeys(self.u, 0.0) # current control input

        self.curr_estate = {}  #: Current state estimate
        self.curr_rstate = {}  #: Current real state
        self.curr_epars = {} #: Current parameter estimate
        self.curr_pstate = {}  #: Current predicted state
        
        self.curr_state_offset = {}  #: Current offset between prediction and real/estimated state
 

        self.xp_l = []
        self.xp_key = {}
 
        # NMPC scheme options
        self.adapt_params = kwargs.pop('adapt_params',False)
        self.robustness_threshold = kwargs.pop('robustness_threshold',0.05)
        self.confidence_threshold = kwargs.pop('confidence_threshold',0.2)
        self.estimate_acceptance = kwargs.pop('estimate_acceptance',1.0e8)
        
        # MHE options
        self.PI_indices = {}
        self._PI = {}
        self._scaled_shape_matrix = None

    """ auxilliary functions """
    def create_tf_bounds(self, m):
        for index in m.tf.index_set():
            m.tf[index].setlb(self.tf_bounds[0])
            m.tf[index].setub(self.tf_bounds[1])
    
    def cycle_iterations(self):
        self._c_it += 1
        self.nfe_mhe += 1
        
    @staticmethod
    def store_results(m):
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

    @staticmethod
    def journalizer(flag, iter, phase, message):
        """Method that writes a little message
        Args:
            flag (str): The flag
            iter (int): The current iteration
            phase (str): The phase
            message (str): The text message to display
        Returns:
            None"""
        iter = str(iter)
        print('-' * 75)
        print('-' * 75)
        if flag == 'W':
            print(flag +'\t'+ iter + '\t' + '[[' + phase + ']]' + ' ' + message, file=sys.stderr)
        else:
            print(flag +'\t'+ iter + '\t' + '[[' + phase + ']]' + ' ' + message)
        print('-' * 75)
        print('-' * 75) 

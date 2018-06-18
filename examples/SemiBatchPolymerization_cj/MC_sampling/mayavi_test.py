#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:44:53 2018

@author: flemmingholtorf
"""
import os
os.environ['QT_API'] = 'pyqt'
# Create the data.
import numpy as np
dimension = 3

kA = np.array([-0.2,0.2])#*e.nominal_parameter_values['kA',()] 
Ap = np.array([-0.2,0.2])#*e.nominal_parameter_values['A',('p',)]
Ai = np.array([-0.2,0.2])#*e.nominal_parameter_values['A',('i',)] 
x = [Ap[0],Ap[1],Ap[0],Ap[1]]  
y = [Ai[0],Ai[1],Ai[0],Ai[0]]
X,Y = np.meshgrid(x,y)
Z_u = np.array([[kA[1],kA[1],kA[1],kA[1]] for i in range(len(x))])
Z_l = np.array([[kA[0],kA[0],kA[0],kA[0]] for i in range(len(x))])
aux = {1:X,2:Y,3:(Z_l,Z_u)}
combinations = [[1,2,3],[1,3,2],[3,1,2]]
facets = {}
b = 0
for combination in combinations:
    facets[b] = np.array([aux[i] if i != 3  else aux[i][0] for i in combination])
    facets[b+1] = np.array([aux[i]  if i != 3  else aux[i][1] for i in combination]) 
    b += 2


s = np.array([5**2,10**2,2.5**2])
radii = 1/np.sqrt(s) # length of half axes, V rotation

# transform in polar coordinates for simpler waz of plotting
u = np.linspace(0.0, 2.0 * np.pi, 30) # angle = idenpendent variable
v = np.linspace(0.0, np.pi, 30) # angle = idenpendent variable
x = radii[0] * np.outer(np.cos(u), np.sin(v)) # x-coordinate
y = radii[1] * np.outer(np.sin(u), np.sin(v)) # y-coordinate
z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
from mayavi import mlab

mlab.mesh(x,y,z,color=(1.0,0,0),opacity=0.5)
for i in facets:
    mlab.mesh(facets[i][0],facets[i][1],facets[i][2],color=(0.0,0.2,0.0),opacity=0.1)
mlab.axes()
mlab.show()
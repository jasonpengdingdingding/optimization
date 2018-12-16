# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:55:50 2018

@author: pasion
"""

import numpy as np
import numpy.linalg as npla

def cqpec_lagrange_multiplier(H,A,c,b):
    '''Convex Quadratic programming with Equality constraints'''
    '''
     min 0.5x'Hx+c'x
     s.t. Ax=b
     
     H (n x n) is symmetric, real, nonsingular and positive definite. 
     A (m x n), the row must be full rank.
     c,x (n x 1). b (m x 1).
    '''
    #the inv function can check that H is symmetric, nosingular, positive definite or not.
    invH = npla.inv(H)
    #if the row of A lacks rank (<m), part1 will be Singular
    part1=npla.multi_dot([A,invH,A.T])
    #print 'part1 dim shape',part1.ndim,part1.shape
    invPart1=npla.inv(part1)
    B=npla.multi_dot([invPart1,A,invH])
    part2=npla.multi_dot([invH,A.T,B])
    G=invH-part2
    C=-invPart1
    x=np.dot(B.T,b)-np.dot(G,c)
    lmbd=np.dot(B,c)-np.dot(C,b)
    val2=npla.multi_dot([x.T,H,x])
    val0=np.multiply(0.5,val2)
    val1=np.dot(c.T,x)
    val=val0+val1
    return x,lmbd,val

def test1():
    H=np.array([[3,-1,0],[-1,2,-1],[0,-1,1]]).reshape(3,3)
    print 'H dim shape',H.ndim,H.shape
    c=np.array([1,1,1]).reshape(3,1)
    print 'c dim shape',c.ndim,c.shape
    A=np.array([1,2,1]).reshape(1,3)
    print 'A dim shape',A.ndim,A.shape
    b=np.array([4]).reshape(1,1)
    print 'b dim shape',b.ndim,b.shape
    x,lmbd,value=cqpec_lagrange_multiplier(H,A,c,b)
    print 'x',x
    print 'lmbd',lmbd
    print 'value',value

def test2():
    H=np.array([[2,-2,0],[-2,4,0],[0,0,2]]).reshape(3,3)
    print 'H dim shape',H.ndim,H.shape
    c=np.array([0,0,1]).reshape(3,1)
    print 'c dim shape',c.ndim,c.shape
    A=np.array([1,1,1,2,-1,1]).reshape(2,3)
    print 'A dim shape',A.ndim,A.shape
    b=np.array([4,2]).reshape(2,1)
    print 'b dim shape',b.ndim,b.shape
    x,lmbd,value=cqpec_lagrange_multiplier(H,A,c,b)
    print 'x',x
    print 'lmbd',lmbd
    print 'value',value

test1()
test2()
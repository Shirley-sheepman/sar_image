# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:13:41 2025

@author: 28211
"""

#import numpy as np
def nextpow2(n):
    k=1
    while(2**k<n):
        k=k+1;
    return k
        

    
    
    
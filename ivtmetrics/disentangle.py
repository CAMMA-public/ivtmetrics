#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An python implementation triplet component filtering .
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
icube, unistra
"""
#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np

#%%%%%%%%% COMPONENT FILTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Disentangle(object):
    """
    Class: filter a triplet prediction into the components (such as instrument i, verb v, target t, instrument-verb iv, instrument-target it, etc)    
    @args
    ----
        url: str. path to the dictionary map file of the dataset decomposition labels            
    @params
    ----------
    bank :   2D array
        holds the dictionary mapping of all components    
    @methods
    ----------
    extract(input, componet): 
        call filter a component labels from the inputs labels     
    """

    def __init__(self, url="./maps.txt"):
        self.bank = np.genfromtxt(url, dtype=int, comments='#', delimiter=',', skip_header=0)
        
    def decompose(self, inputs, component):
        """ Extract the component labels from the triplets.
            @args:
                inputs: a 1D vector of dimension (n), where n = number of triplet classes;
                        with values int(0 or 1) for target labels and float[0, 1] for predicted labels.
                component: a string for the component to extract; 
                        (e.g.: i for instrument, v for verb, t for target, iv for instrument-verb pair, it for instrument-target pair and vt (unused) for verb-target pair)
            @return:
                output: int or float sparse encoding 1D vector of dimension (n), where n = number of component's classes.
        """
        txt2id = {'ivt':0, 'i':1, 'v':2, 't':3, 'iv':4, 'it':5, 'vt':6} 
        key    = txt2id[component]
        index  = sorted(np.unique(self.bank[:,key]))
        output = []
        for idx in index:
            same_class  = [i for i,x in enumerate(self.bank[:,key]) if x==idx]
            y           = np.max(np.array(inputs[same_class]))
            output.append(y)        
        return output
    
    def extract(self, inputs, component="i"):
        """
        Extract a component label from the triplet label
        @args
        ----
        inputs: 2D array,
            triplet labels, either predicted label or the groundtruth
        component: str,
            the symbol of the component to extract, choose from
            i: instrument
            v: verb
            t: target
            iv: instrument-verb
            it: instrument-target
            vt: verb-target (not useful)
        @return
        ------
        label: 2D array,
            filtered component's labels of the same shape and data type as the inputs
        """      
        if component == "ivt":
            return inputs
        else:
            component = [component]* len(inputs)
            return np.array(list(map(self.decompose, inputs, component)))

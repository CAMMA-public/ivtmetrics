#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An python implementation recognition AP for surgical action triplet evaluation.
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
icube, unistra
"""
#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from sklearn.metrics import average_precision_score
import warnings
import sys
from ivtmetrics.disentangle import Disentangle

#%%%%%%%%%% RECOGNITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Recognition(Disentangle):
    """
    Class: compute (mean) Average Precision    
    @args
    ----
        num_class: int, optional. The number of class of the classification task (default = 100)            
    @attributes
    ----------
    predictions:    2D array
        holds the accumulated predictions before a reset()
    targets:        2D array
        holds the accumulated groundtruths before a reset()    
    @methods
    -------
    GENERIC
    ------- 
    reset(): 
        call at the beginning of new experiment or epoch to reset all accumulators.
    update(targets, predictions): 
        call per iteration to update the class accumulators for predictions and corresponding groundtruths.   
    video_end(): 
        call at the end of every video during inference to log performance per video.
        
    RESULTS
    ----------
    compute_AP(): 
        call at any point to check the performance of all seen examples after the last reset() call.
    compute_video_AP(): 
        call at any time, usually at the end of experiment or inference, to obtain the performance of all tested videos.  
    compute_global_AP(): 
        call at any point, compute the framewise AP for all frames across all videos and mAP      
    compute_per_video_mAP(self):
        show mAP per video (not very useful)
    topk(k):
        obtain top k=[5,10,20, etc] performance
    topClass(k):
        obtain top-k correctly detected classes      
    """    
    def __init__(self, num_class=100):
        super(Recognition, self).__init__()
        np.seterr(divide='ignore', invalid='ignore')
        self.num_class = num_class
        self.reset_global()   
        
    ##%%%%%%%%%%%%%%%%%%% RESET OP #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def reset(self):
        "call at the beginning of new experiment or epoch to reset the accumulators for preditions and groundtruths."
        self.predictions = np.empty(shape = [0,self.num_class], dtype=np.float)
        self.targets     = np.empty(shape = [0,self.num_class], dtype=np.int)        
        
    def reset_global(self):
        "call at the beginning of new experiment"
        self.global_predictions = []
        self.global_targets     = []
        self.reset()    
    
    ##%%%%%%%%%%%%%%%%%%% UPDATE OP #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    def update(self, targets, predictions):
        """
        update prediction function
        @args
        -----
        targets: 2D array, float
            groundtruth of shape (F, C) where F = number of frames, C = number of class
        predictions: 2D array, int
            model prediction of the shape as the groundtruth
        """
        self.predictions = np.append(self.predictions, predictions, axis=0)
        self.targets     = np.append(self.targets, targets, axis=0)      
        
    def video_end(self):
        "call to signal the end of current video. Needed during inference to log performance per video"        
        self.global_predictions.append(self.predictions)
        self.global_targets.append(self.targets)
        self.reset()
    
    ##%%%%%%%%%%%%%%%%%%% COMPUTE OP #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def compute_AP(self, component="ivt"):
        """
        compute performance for all seen examples after a reset()
        @args
        ----
        component: str (optional) default: ivt for triplets
            a str for the component of interest. i for instruments, v for verbs, t for targets, iv for instrument-verb, it for instrument-target, ivt for instrument-verb-target
        @return
        -------
        classwise: 1D array, float
            AP performance per class
        mean: float
            mean AP performance
        """
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets  = self.extract(self.targets, component)
            predicts = self.extract(self.predictions, component)
        else:
            sys.exit(f"Function filtering {component} not supported yet!")
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')            
            classwise = average_precision_score(targets, predicts, average=None)
            mean      = np.nanmean(classwise)
        return {"AP":classwise, "mAP":mean}
    
    def compute_global_AP(self, component="ivt"):
        """
        compute performance for all seen examples after a reset_global()
        @args
        ----
        component: str (optional) default: ivt for triplets
            a str for the component of interest. i for instruments, v for verbs, t for targets, iv for instrument-verb, it for instrument-target, ivt for instrument-verb-target
        @return
        -------
        classwise: 1D array, float
            AP performance per class
        mean: float
            mean AP performance
        """        
        global_targets      = self.global_targets
        global_predictions  = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        targets  = np.concatenate(global_targets, axis=0)
        predicts = np.concatenate(global_predictions, axis=0)
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets  = self.extract(targets, component)
            predicts = self.extract(predicts, component)
        else:
            sys.exit(f"Function filtering {component} not supported yet!")            
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')            
            classwise = average_precision_score(targets, predicts, average=None)
            mean      = np.nanmean(classwise)
        return {"AP":classwise, "mAP":mean}    
    
    def compute_video_AP(self, component="ivt"):
        """
        compute performance video-wise AP
        @args
        ----
        component: str (optional) default: ivt for triplets
            a str for the component of interest. i for instruments, v for verbs, t for targets, iv for instrument-verb, it for instrument-target, ivt for instrument-verb-target
        @return
        -------
        classwise: 1D array, float
            AP performance per class for all videos
        mean: float
            mean AP performance for all videos
        """           
        global_targets      = self.global_targets
        global_predictions  = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        video_log = []
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            for targets, predicts in zip(global_targets, global_predictions):
                if component in ["ivt", "it", "iv", "t", "v", "i"]:
                    targets  = self.extract(targets, component)
                    predicts = self.extract(predicts, component)
                else:
                    sys.exit(f"Function filtering {component} not supported yet!")                        
                classwise = average_precision_score(targets, predicts, average=None)
                video_log.append( classwise.reshape([1,-1]) )
        video_log = np.concatenate(video_log, axis=0)         
        videowise = np.nanmean(video_log, axis=0)
        mean      = np.nanmean(videowise)
        return {"AP":classwise, "mAP":mean}

    ##%%%%%%%%%%%%%%%%%%% TOP OP #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    def topK(self, k=5, component="ivt"):
        """
        compute topK performance for all seen examples after a reset_global()        
        @args
        ----
        k: int
            number of chances of correct prediction
        component: str (optional) default: ivt for triplets
            a str for the component of interest. i for instruments, v for verbs, t for targets, iv for instrument-verb, it for instrument-target, ivt for instrument-verb-target.            
        @return
        ----
        mean: float
            mean top-k performance
        """            
        global_targets      = self.global_targets
        global_predictions  = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        targets  = np.concatenate(global_targets, axis=0)
        predicts = np.concatenate(global_predictions, axis=0)
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets  = self.extract(targets, component)
            predicts = self.extract(predicts, component)
        else:
            sys.exit(f"Function filtering {component} not supported yet!")
        correct = 0.0
        total   = 0
        for gt, pd in zip(targets, predicts):
            gt_pos  = np.nonzero(gt)[0]
            pd_idx  = (-pd).argsort()[:k]
            correct += len(set(gt_pos).intersection(set(pd_idx)))
            total   += len(gt_pos)
        if total==0: total=1
        return correct/total

    def topClass(self, k=10, component="ivt"):
        """
        compute top K recognize classes for all seen examples after a reset_global()        
        @args
        ----
        k: int
            number of chances of correct prediction            
        @return
        ----
        mean: float
            mean top-k recognized classes
        """
        global_targets      = self.global_targets
        global_predictions  = self.global_predictions
        if len(self.targets) > 0:
            global_targets.append(self.targets)
            global_predictions.append(self.predictions)
        targets  = np.concatenate(global_targets, axis=0)
        predicts = np.concatenate(global_predictions, axis=0)
        if component in ["ivt", "it", "iv", "t", "v", "i"]:
            targets  = self.extract(targets, component)
            predicts = self.extract(predicts, component)
        else:
            sys.exit(f"Function filtering {component} not supported yet!")            
        classwise = average_precision_score(targets, predicts, average=None)
        pd_idx    = (-classwise).argsort()[:k]
        output    = {x:classwise[x] for x in pd_idx}
        return output
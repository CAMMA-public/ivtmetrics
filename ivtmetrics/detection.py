#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An python implementation detection AP metrics for surgical action triplet evaluation.
Created on Thu Dec 30 12:37:56 2021
@author: nwoye chinedu i.
camma, ihu, icube, unistra, france
"""
#%%%%%%%% imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import sys
import warnings
from ivtmetrics.recognition import Recognition

#%%%%%%%%%% DETECTION AND ASSOCIATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Detection(Recognition):
    """
    @args: 
        init(num_class, num_tool, num_target, threshold):
            num_class: number of triplet categories
            num_tool: number of tool categories
            num_target: number of target categories
            threshold: IoU overlap threshold score
        
    @call: 
        update(targets, predictions):
            targets: groundtruth json file with "frameid" as key
            predictions: prediction json file with "frameid" as key
            json format:
                frameid1: {
                            "recognition": List of class-wise probabilities in the 
                                format:  [score1, score2, score3, ..., score100],
                                example: [0.001, 0.025, 0.911, ...].
                            "detection" List of list of box detection for each triplet in the 
                                format:  [[class1,score,x,y,w,h], [class3,score,x,y,w,h], [class56,score,x,y,w,h], ...],
                                example: [[1,0.842,0.21,0.09,0.32,0.33], [3,0.992,0.21,0.09,0.32,0.33], [56,0.422,0.21,0.09,0.32,0.33], ....] 
                         }
                frameid2: {...}
                    :
        extract(input, componet): filter a component labels from the inputs labels
        compute_AP('i/ivt') return AP for current run
        compute_video_AP('i/ivt') return AP from video-wise averaging
        compute_global_AP('i/ivt') return AP for all seen examples
        reset_video()
          
    @return     
        output:
            detection and association performances:
                AP: average precision
                mAP: mean average precision
                Rec: Recall
                mRec: Mean Recall
                Pre: Precision
                mPrec: Mean Precision
                lm: localize and match percent
                plm: partially localize and match
                ids: identity switch
                idm: identity miss
                mil: missed localization
                fp: false positives
                fn: false negatives    
    @params
    --------  
    @methods
    -------
    @format
        box format: [{"triplet":tid, "instrument":[tool, 1.0, x,y,w,h], "target":[]}]
    """
    def __init__(self, num_class=100, num_tool=6, num_target=15, threshold=0.5):
        super(Recognition, self).__init__()
        self.num_class      = num_class  
        self.num_tool       = num_tool                
        self.classwise_ap   = []
        self.classwise_rec  = []
        self.classwise_prec = []
        self.accumulator    = {}
        self.video_count    = 0
        self.end_call       = False
        self.threshold      = threshold
        self.reset()        
                
    def reset(self):
        self.video_count = 0
        self.video_end()  
        
    def reset_global(self):
        self.video_count = 0
        self.video_end()    
        
    def video_end(self):
        self.video_count += 1
        self.end_call = True
        self.accumulator[self.video_count] = {
            "hits":  [[] for _ in range(self.num_class)],
            "ndet":  [0  for _ in range(self.num_class)],
            "npos":  [0  for _ in range(self.num_class)],                             
            "hits_i":[[] for _ in range(self.num_tool)],
            "ndet_i":[0  for _ in range(self.num_tool)] ,
            "npos_i":[0  for _ in range(self.num_tool)] ,                    
            "fp": 0,
            "fn": 0,
            "lm": 0,
            "plm": 0,
            "ids": 0,
            "idm": 0,
            "mil": 0,
        }
    
    def xywh2xyxy(self, bb):
        bb[2] += bb[0]
        bb[3] += bb[1]
        return bb    
    
    def iou(self, bb1, bb2):
        bb1 = self.xywh2xyxy(bb1)
        bb2 = self.xywh2xyxy(bb2)
        x1 = bb1[2] - bb1[0]
        y1 = bb1[3] - bb1[1]
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        x2 = bb2[2] - bb2[0]
        y2 = bb2[3] - bb2[1]
        if x2 < 0: x2 = 0
        if y2 < 0: y2 = 0
        xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
        yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
        if xiou < 0: xiou = 0
        if yiou < 0: yiou = 0
        if xiou * yiou <= 0:
            return 0
        else:
            return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)        
        
    def is_match(self, det_gt, det_pd, threshold):  
        if det_gt[0] == det_pd[0]: # cond 1: correct identity        
            if self.iou(det_gt[-4:], det_pd[-4:]) >= threshold: # cond 2: sufficient iou
                return True
        return False 
    
    def is_partial_match(self, det_gt, det_pd):  
        if det_gt[0] == det_pd[0]: # cond 1: correct identity        
            if self.iou(det_gt[-4:], det_pd[-4:]) > 0.0: # cond 2: insufficient iou
                return True
        return False
        
    def is_id_switch(self, det_gt, det_pd, det_gts, threshold):        
        if self.iou(det_gt[-4:], det_pd[-4:]) > threshold: # cond 2: insufficient/sufficient iou 
            gt_ids = list(det_gts[:,0])
            if det_pd[0] in gt_ids: # cond 1: switched identity
                return np.where(gt_ids==det_pd[0])[0][0]
        return False    
    
    def is_id_miss(self, det_gt, det_pd, threshold):        
        if self.iou(det_gt[-4:], det_pd[-4:]) > threshold: # cond 2: insufficient/sufficient iou 
                return True
        return False
        
    def is_miss_loc(self, det_gt, det_pd, det_gts):        
        gt_ids = list(det_gts[:,0])
        if det_pd[0] in gt_ids: # cond 1: correct identity only
            return np.where(gt_ids==det_pd[0])[0][0]
        return False   
    
    def separate_detection(self, det_gts, det_pds):
        pos_ids = list(det_gts[:,0])
        matching_dets   = [list(x) for x in det_pds if x[0] in pos_ids]
        unmatching_dets = [list(x) for x in det_pds if x[0] not in pos_ids]
        return matching_dets, unmatching_dets
    
    def list2stack(self, x):
        # if x == []: x = [[-1,-1,-1,-1,-1,-1]] # empty
        if x == []: x = [[]] # empty
        #x format for a single frame: list(list): each list = [tripletID, toolID, toolProbs, x, y, w, h] bbox is scaled (0..1)
        assert isinstance(x[0], list), "Each frame must be a list of lists, each list a prediction of triplet and object locations"        
        if len(x[0]):
            x = np.stack(x, axis=0)
            x = x[x[:,2].argsort()[::-1]]
        return x    
    
    def sortstack(self, x):
        #x format for a single frame: list(list): each list = [tripletID, toolID, toolProbs, x, y, w, h] bbox is scaled (0..1)
        assert isinstance(x, np.ndarray), "Each frame must be an n-dim array with each row a unique prediction of triplet and object locations"
        x = x[x[:,2].argsort()[::-1]]
        return x
        
    def dict2stack(self, x):
        #x format for a single frame: list(dict): each dict = {"triplet":ID, "instrument": [ID, Probs, x, y, w, h]} bbox is scaled (0..1)
        assert isinstance(x, list), "Each frame must be a list of dictionaries"        
        y = []
        for d in x:
            assert isinstance(d, dict), "Each frame must be a list of dictionaries, each dictionary a prediction of triplet and object locations"
            p = [d['triplet']]
            p.extend(d["instrument"])
            y.append(p)
        return self.list2stack(y)    
    
    def update(self, targets, predictions, format="list"): 
        [self.update_frame(y, f, format) for y,f in zip(targets, predictions)]
#        print("First")
#        formats = [format]* len(targets)
#        map(self.update_frame, targets, predictions, formats)  
#        for item in range(len(targets)):
#            self.update_frame(targets[item], predictions[item], format)
        self.end_call = False 
    
    def update_frame(self, targets, predictions, format="list"):
        if format=="list":            
            detection_gt    = self.list2stack(targets)
            detection_pd    = self.list2stack(predictions)
        elif format=="dict":
            detection_gt    = self.dict2stack(targets)
            detection_pd    = self.dict2stack(predictions)
        else:
            sys.exit("unkown input format for update function. Must be a list or dict")
        if len(detection_pd[0]) + len(detection_gt[0]) == 0:
            return
        detection_gt_ivt = detection_gt.copy()
        detection_pd_ivt = detection_pd.copy()
        # for triplet        
        for gt in detection_gt_ivt: 
            if len(gt): self.accumulator[self.video_count]["npos"][int(gt[0])] += 1
        for det_pd in detection_pd_ivt:
            if len(det_pd):
                self.accumulator[self.video_count]["ndet"][int(det_pd[0])] += 1
                matched = False                
                for k, det_gt in enumerate(detection_gt_ivt):
                    if len(det_gt): 
                        y = det_gt[0:] 
                        f = det_pd[0:]
                        if self.is_match(y, f, threshold=self.threshold):
                            detection_gt_ivt = np.delete(detection_gt_ivt, obj=k, axis=0)
                            matched = True
                            break
                if matched:
                    self.accumulator[self.video_count]["hits"][int(det_pd[0])].append(1.0)
                else:
                    self.accumulator[self.video_count]["hits"][int(det_pd[0])].append(0.0)
        # for instrument       
        detection_gt_i = detection_gt.copy()
        detection_pd_i = detection_pd.copy() 
        for gt in detection_gt_i:
            if len(gt): self.accumulator[self.video_count]["npos_i"][int(gt[1])] += 1        
        for det_pd in detection_pd_i:
            if len(det_pd):
                self.accumulator[self.video_count]["ndet_i"][int(det_pd[1])] += 1
                matched = False                
                for k, det_gt in enumerate(detection_gt_i): 
                    if len(det_gt): 
                        y = det_gt[1:] 
                        f = det_pd[1:]
                        if self.is_match(y, f, threshold=self.threshold):
                            detection_gt_i = np.delete(detection_gt_i, obj=k, axis=0)
                            matched = True
                            break
                if matched:
                    self.accumulator[self.video_count]["hits_i"][int(det_pd[1])].append(1.0)
                else:
                    self.accumulator[self.video_count]["hits_i"][int(det_pd[1])].append(0.0)  
        # process association
        self.association(targets=detection_gt.copy(), predictions=detection_pd.copy())
        
    def association(self, targets, predictions): 
        detection_gt = targets.copy()
        detection_pd = predictions.copy()
        if len(detection_gt[0])==0:
            self.accumulator[self.video_count]["fp"] += len([x for x in detection_pd if len(x)])
        elif len(detection_pd[0])==0:    
            self.accumulator[self.video_count]["fn"] += len([x for x in detection_gt if len(x)])
        else:
            # separate
            matched_dets, unmatched_dets = self.separate_detection(detection_gt, detection_pd)
            # compare
            matched_dets, detection_gt  = self.localized_box_matched_id(matched_dets, detection_gt)
            matched_dets, detection_gt  = self.partially_localized_box_matched_id(matched_dets, detection_gt)
            matched_dets, detection_gt  = self.localized_box_switched_id(matched_dets, detection_gt)
            matched_dets, detection_gt  = self.localized_box_missed_id(matched_dets, unmatched_dets, detection_gt)
            matched_dets, detection_gt  = self.unlocalized_box_matched_id(matched_dets, detection_gt)        
            # False positives and negatives
            self.accumulator[self.video_count]["fp"] += len([x for x in matched_dets if len(x)])
            self.accumulator[self.video_count]["fn"] += len([x for x in detection_gt if len(x)])
        return
    
    def localized_box_matched_id(self, matched_dets, detection_gt):
        # LM: fully localized and matched
        leftover = []
        if len(matched_dets):
            for det_pd in matched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y = det_gt[0:]
                    if self.is_match(y, f, threshold=self.threshold):
                        detection_gt = np.delete(detection_gt, obj=k, axis=0)
                        matched = True
                        break                
                if matched:
                    self.accumulator[self.video_count]["lm"] += 1
                else:
                    leftover.append(det_pd)
        matched_dets = leftover.copy()
        return matched_dets, detection_gt            
            
    def partially_localized_box_matched_id(self, matched_dets, detection_gt):
        # pLM: partially localized and matched
        leftover = []
        if len(matched_dets):
            for det_pd in matched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y = det_gt[0:]
                    if self.is_partial_match(y, f):
                        detection_gt = np.delete(detection_gt, obj=k, axis=0)
                        matched = True
                        break                
                if matched:
                    self.accumulator[self.video_count]["plm"] += 1
                else:
                    leftover.append(det_pd)
        matched_dets = leftover.copy()
        return matched_dets, detection_gt        
        
    def localized_box_switched_id(self, matched_dets, detection_gt):
        # IDS: partially localized but identity switched
        leftover = []
        if len(matched_dets):
            for det_pd in matched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y   = det_gt[0:]
                    ids = self.is_id_switch(y, f, detection_gt, threshold=self.threshold)
                    if ids:
                        detection_gt = np.delete(detection_gt, obj=ids, axis=0)
                        matched = True
                        break  
                if matched:
                    self.accumulator[self.video_count]["ids"] += 1
                else:
                    leftover.append(det_pd)  
        matched_dets = leftover.copy()
        return matched_dets, detection_gt                
                
    def localized_box_missed_id(self, matched_dets, unmatched_dets, detection_gt):
        # IDS: partially localized but identity missed
        unmatched_dets += matched_dets
        leftover = []
        if len(matched_dets):
            for det_pd in unmatched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y   = det_gt[0:]
                    if self.is_id_miss(y, f, threshold=self.threshold):
                        matched = True
                        break  
                if matched:
                    self.accumulator[self.video_count]["idm"] += 1
                else:
                    leftover.append(det_pd)  
        matched_dets = leftover.copy()
        return matched_dets, detection_gt
        
    def unlocalized_box_matched_id(self, matched_dets, detection_gt):
        # IDS: partially localized but identity switched
        leftover = []
        if len(matched_dets):
            for det_pd in matched_dets: 
                f = det_pd[0:]
                matched = False
                # if len(detection_gt[0]):
                for k, det_gt in enumerate(detection_gt):
                    y   = det_gt[0:]
                    ids = self.is_miss_loc(y, f, detection_gt)
                    if ids:
                        detection_gt = np.delete(detection_gt, obj=ids, axis=0)
                        matched = True
                        break  
                if matched:
                    self.accumulator[self.video_count]["mil"] += 1
                else:
                    leftover.append(det_pd)  
        matched_dets = leftover.copy()
        return matched_dets, detection_gt        
        
    def eval_association(self, accumulator):
        fp    = accumulator['fp']
        fn    = accumulator['fn']
        lm    = accumulator['lm']
        plm   = accumulator['plm']
        ids   = accumulator['ids']
        idm   = accumulator['idm']
        mil  = accumulator['mil']
        total = fp + fn + lm + plm + ids + idm + mil
        if total==0: 
            return [np.nan]*7
        fp   = fp/total
        fn   = fn/total
        lm   = lm/total
        plm  = plm/total
        ids  = ids/total
        idm  = idm/total
        mil = mil/total
        return (lm, plm, ids, idm, mil, fp, fn)        
                        
    def compute(self, component="ivt", video_id=None):
        classwise_ap    = []
        classwise_rec   = []
        classwise_prec  = []
        if video_id == None: 
            video_id = self.video_count-1 if self.end_call else self.video_count
        hit_str     = "hits" if component=="ivt" else "hits_i"
        pos_str     = "npos" if component=="ivt" else "npos_i"
        det_str     = "ndet" if component=="ivt" else "ndet_i"
        num_class   = self.num_class if component=="ivt" else self.num_tool       
        # decide on accumulator for framewise / video wise / current
        if video_id == -1:
            accumulator = {}
            accumulator[hit_str] = [sum([p[k]for p in [self.accumulator[f][hit_str] for f in self.accumulator] ],[]) for k in range(num_class)]            
            accumulator[pos_str] = list(np.sum(np.stack([self.accumulator[f][pos_str] for f in self.accumulator]), axis=0))       
            accumulator[det_str] = list(np.sum(np.stack([self.accumulator[f][det_str] for f in self.accumulator]), axis=0))
        else:
             accumulator = self.accumulator[video_id]        
        # compuatation
        for hits, npos, ndet in zip(accumulator[hit_str], accumulator[pos_str], accumulator[det_str]): # loop for num_class 
            if npos + ndet == 0: # no gt instance and no detection for the class
                classwise_ap.append(np.nan)
                classwise_rec.append(np.nan)
                classwise_prec.append(np.nan)
            elif npos>0 and len(hits)==0: # no detections but there are gt instances for the class
                classwise_ap.append(0.0)
                classwise_rec.append(0.0)
                classwise_prec.append(0.0)
            else:
                hits = np.cumsum(hits)
                ap   = 0.0
                rec  = hits / npos if npos else 0.0
                prec = hits / (np.array(range(len(hits)), dtype=float) + 1.0)
                for i in range(11):
                    mask = rec >= (i / 10.0)
                    if np.sum(mask) > 0:
                        ap += np.max(prec[mask]) / 11.0
                classwise_ap.append(ap)
                classwise_rec.append(np.max(rec))
                classwise_prec.append(np.max(prec))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            assoc_results = self.eval_association(accumulator)
            return (classwise_ap, np.nanmean(classwise_ap)), \
                    (classwise_rec, np.nanmean(classwise_rec)), \
                    (classwise_prec, np.nanmean(classwise_prec)), \
                    assoc_results
    
    def compute_video_AP(self, component="ivt"):
        classwise_ap    = []
        classwise_rec   = []
        classwise_prec  = []
        video_lm, video_plm, video_ids, video_idm, video_mil, video_fp, video_fn = [],[],[],[],[],[],[]
        for j in range(self.video_count):
            video_id = j+1
            (ap, _), (rec, _), (prec, _), asc = self.compute(component=component, video_id=video_id)            
            classwise_ap.append(ap)
            classwise_rec.append(rec)
            classwise_prec.append(prec)
            video_lm.append(asc[0])  # association metrics starts
            video_plm.append(asc[1])
            video_ids.append(asc[2])
            video_idm.append(asc[3])
            video_mil.append(asc[4])
            video_fp.append(asc[5])
            video_fn.append(asc[6])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            classwise_ap    = np.nanmean(np.stack(classwise_ap, axis=0), axis=0)
            classwise_rec   = np.nanmean(np.stack(classwise_rec, axis=0), axis=0)
            classwise_prec  = np.nanmean(np.stack(classwise_prec, axis=0), axis=0)        
            mAP             = np.nanmean(classwise_ap)
            mRec            = np.nanmean(classwise_rec)
            mPrec           = np.nanmean(classwise_prec) 
            lm              = np.nanmean(video_lm)  # association metrics starts
            plm             = np.nanmean(video_plm)
            ids             = np.nanmean(video_ids)
            idm             = np.nanmean(video_idm)
            mil            = np.nanmean(video_mil)
            fp              = np.nanmean(video_fp)
            fn              = np.nanmean(video_fn)            
        return {
                "AP":classwise_ap, "mAP":mAP, "Rec":classwise_rec, "mRec":mRec, "Pre":classwise_prec, "mPre":mPrec,
                "lm":lm, "plm":plm, "ids":ids, "idm":idm, "mil":mil, "fp":fp, "fn":fn,
               }
    
    def compute_AP(self, component="ivt"):
        a,r,p, asc = self.compute(component=component, video_id=None)
        (lm, plm, ids, idm, mil, fp, fn) = asc
        return {"AP":a[0], "mAP":a[1], "Rec":r[0], "mRec":r[1], "Pre":p[0], "mPre":p[1],
                "lm":lm, "plm":plm, "ids":ids, "idm":idm, "mil":mil, "fp":fp, "fn":fn,}
        
    def compute_global_AP(self, component="ivt"):
        a,r,p, asc =  self.compute(component=component, video_id=-1)
        (lm, plm, ids, idm, mil, fp, fn) = asc
        return {"AP":a[0], "mAP":a[1], "Rec":r[0], "mRec":r[1], "Pre":p[0], "mPre":p[1],
                "lm":lm, "plm":plm, "ids":ids, "idm":idm, "mil":mil, "fp":fp, "fn":fn,}
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from sacred import Experiment
from model.config import cfg as frcnn_cfg
import os
import os.path as osp
import yaml
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt

from tracker.rfrcnn import FRCNN as rFRCNN
from tracker.vfrcnn import FRCNN as vFRCNN
from tracker.config import cfg, get_output_dir
from tracker.utils import plot_sequence
from tracker.mot_sequence import MOT_Sequence
from tracker.kitti_sequence import KITTI_Sequence
from tracker.datasets.factory import Datasets
from tracker.tracker_debug import Tracker
from tracker.utils import interpolate
from tracker.resnet import resnet50

from sklearn.utils.linear_assignment_ import linear_assignment
from easydict import EasyDict as edict
from mot_evaluation.io import read_txt_to_struct, read_seqmaps, extract_valid_gt_data, print_metrics
from mot_evaluation.bbox import bbox_overlap
from mot_evaluation.measurements import clear_mot_hungarian, idmeasures

ex = Experiment()

def preprocessingDB(trackDB, gtDB, distractor_ids, iou_thres, minvis):
    """
    Preprocess the computed trajectory data.
    Matching computed boxes to groundtruth to remove distractors and low visibility data in both trackDB and gtDB
    trackDB: [npoints, 9] computed trajectory data
    gtDB: [npoints, 9] computed trajectory data
    distractor_ids: identities of distractors of the sequence
    iou_thres: bounding box overlap threshold
    minvis: minimum visibility of groundtruth boxes, default set to zero because the occluded people are supposed to be interpolated for tracking.
    """
    track_frames = np.unique(trackDB[:, 0])
    gt_frames = np.unique(gtDB[:, 0])
    nframes = min(len(track_frames), len(gt_frames))  
    res_keep = np.ones((trackDB.shape[0], ), dtype=float)
    for i in range(1, nframes + 1):
        # find all result boxes in this frame
        res_in_frame = np.where(trackDB[:, 0] == i)[0]
        res_in_frame_data = trackDB[res_in_frame, :]
        gt_in_frame = np.where(gtDB[:, 0] == i)[0]
        gt_in_frame_data = gtDB[gt_in_frame, :]
        res_num = res_in_frame.shape[0]
        gt_num = gt_in_frame.shape[0]
        overlaps = np.zeros((res_num, gt_num), dtype=float)
        for gid in range(gt_num):
            overlaps[:, gid] = bbox_overlap(res_in_frame_data[:, 2:6], gt_in_frame_data[gid, 2:6]) 
        matched_indices = linear_assignment(1 - overlaps)
        for matched in matched_indices:
            # overlap lower than threshold, discard the pair
            if overlaps[matched[0], matched[1]] < iou_thres:
                continue

            # matched to distractors, discard the result box
            if gt_in_frame_data[matched[1], 1] in distractor_ids:
                res_keep[res_in_frame[matched[0]]] = 0
            
            # matched to a partial
            if gt_in_frame_data[matched[1], 8] < minvis:
                res_keep[res_in_frame[matched[0]]] = 0
            

        # sanity check
        frame_id_pairs = res_in_frame_data[:, :2]
        uniq_frame_id_pairs = np.unique(frame_id_pairs)
        has_duplicates = uniq_frame_id_pairs.shape[0] < frame_id_pairs.shape[0]
        assert not has_duplicates, 'Duplicate ID in same frame [Frame ID: %d].'%i
    keep_idx = np.where(res_keep == 1)[0]
    #print('[TRACK PREPROCESSING]: remove distractors and low visibility boxes, remaining %d/%d computed boxes'%(len(keep_idx), len(res_keep)))
    trackDB = trackDB[keep_idx, :]
    #print('Distractors:', distractor_ids)
    #keep_idx = np.array([i for i in xrange(gtDB.shape[0]) if gtDB[i, 1] not in distractor_ids and gtDB[i, 8] >= minvis])
    keep_idx = np.array([i for i in range(gtDB.shape[0]) if gtDB[i, 6] != 0])
    #print('[GT PREPROCESSING]: Removing distractor boxes, remaining %d/%d computed boxes'%(len(keep_idx), gtDB.shape[0]))
    gtDB = gtDB[keep_idx, :]
    return trackDB, gtDB


def evaluate_sequence(trackDB, gtDB, distractor_ids, iou_thres=0.5, minvis=0):
    """
    Evaluate single sequence
    trackDB: tracking result data structure
    gtDB: ground-truth data structure
    iou_thres: bounding box overlap threshold
    minvis: minimum tolerent visibility
    """
    trackDB, gtDB = preprocessingDB(trackDB, gtDB, distractor_ids, iou_thres, minvis)
    mme, c, fp, g, missed, d, M, allfps, clear_mot_info = clear_mot_hungarian(trackDB, gtDB, iou_thres)
    #print(mme)
    #print(c)
    #print(fp)
    #print(g)

    gt_frames = np.unique(gtDB[:, 0])
    gt_ids = np.unique(gtDB[:, 1])
    st_ids = np.unique(trackDB[:, 1])
    f_gt = len(gt_frames)
    n_gt = len(gt_ids)
    n_st = len(st_ids)

    FN = sum(missed)
    FP = sum(fp)
    IDS = sum(mme)
    MOTP = (sum(sum(d)) / sum(c)) * 100                                                 # MOTP = sum(iou) / # corrected boxes
    MOTAL = (1 - (sum(fp) + sum(missed) + np.log10(sum(mme) + 1)) / sum(g)) * 100       # MOTAL = 1 - (# fp + # fn + #log10(ids)) / # gts
    MOTA = (1 - (sum(fp) + sum(missed) + sum(mme)) / sum(g)) * 100                      # MOTA = 1 - (# fp + # fn + # ids) / # gts
    recall = sum(c) / sum(g) * 100                                                      # recall = TP / (TP + FN) = # corrected boxes / # gt boxes
    precision = sum(c) / (sum(fp) + sum(c)) * 100                                       # precision = TP / (TP + FP) = # corrected boxes / # det boxes
    FAR = sum(fp) / f_gt                                                                # FAR = sum(fp) / # frames
    MT_stats = np.zeros((n_gt, ), dtype=float)
    for i in range(n_gt):
        gt_in_person = np.where(gtDB[:, 1] == gt_ids[i])[0]
        gt_total_len = len(gt_in_person)
        gt_frames_tmp = gtDB[gt_in_person, 0].astype(int)
        gt_frames_list = list(gt_frames)
        st_total_len = sum([1 if i in M[gt_frames_list.index(f)].keys() else 0 for f in gt_frames_tmp])
        ratio = float(st_total_len) / gt_total_len
        
        if ratio < 0.2:
            MT_stats[i] = 1
        elif ratio >= 0.8:
            MT_stats[i] = 3
        else:
            MT_stats[i] = 2
            
    ML = len(np.where(MT_stats == 1)[0])
    PT = len(np.where(MT_stats == 2)[0])
    MT = len(np.where(MT_stats == 3)[0])

    # fragment
    fr = np.zeros((n_gt, ), dtype=int)
    M_arr = np.zeros((f_gt, n_gt), dtype=int)
    
    for i in range(f_gt):
        for gid in M[i].keys():
            M_arr[i, gid] = M[i][gid] + 1
    
    for i in range(n_gt):
        occur = np.where(M_arr[:, i] > 0)[0]
        occur = np.where(np.diff(occur) != 1)[0]
        fr[i] = len(occur)
    FRA = sum(fr)
    idmetrics = idmeasures(gtDB, trackDB, iou_thres)
    metrics = [idmetrics.IDF1, idmetrics.IDP, idmetrics.IDR, recall, precision, FAR, n_gt, MT, PT, ML, FP, FN, IDS, FRA, MOTA, MOTP, MOTAL]
    extra_info = edict()
    extra_info.mme = sum(mme)
    extra_info.c = sum(c)
    extra_info.fp = sum(fp)
    extra_info.g = sum(g)
    extra_info.missed = sum(missed)
    extra_info.d = d
    #extra_info.m = M
    extra_info.f_gt = f_gt
    extra_info.n_gt = n_gt
    extra_info.n_st = n_st
#    extra_info.allfps = allfps

    extra_info.ML = ML
    extra_info.PT = PT
    extra_info.MT = MT
    extra_info.FRA = FRA
    extra_info.idmetrics = idmetrics

    ML_PT_MT = [gt_ids[np.where(MT_stats == 1)[0]], gt_ids[np.where(MT_stats == 2)[0]], gt_ids[np.where(MT_stats == 3)[0]]]

    return metrics, extra_info, clear_mot_info, ML_PT_MT, M

   

def evaluate_bm(all_metrics):
    """
    Evaluate whole benchmark, summaries all metrics
    """
    f_gt, n_gt, n_st = 0, 0, 0
    nbox_gt, nbox_st = 0, 0
    c, g, fp, missed, ids = 0, 0, 0, 0, 0
    IDTP, IDFP, IDFN = 0, 0, 0
    MT, ML, PT, FRA = 0, 0, 0, 0
    overlap_sum = 0
    for i in range(len(all_metrics)):
        nbox_gt += all_metrics[i].idmetrics.nbox_gt
        nbox_st += all_metrics[i].idmetrics.nbox_st
        # Total ID Measures
        IDTP += all_metrics[i].idmetrics.IDTP
        IDFP += all_metrics[i].idmetrics.IDFP
        IDFN += all_metrics[i].idmetrics.IDFN
        # Total ID Measures
        MT += all_metrics[i].MT 
        ML += all_metrics[i].ML
        PT += all_metrics[i].PT 
        FRA += all_metrics[i].FRA 
        f_gt += all_metrics[i].f_gt 
        n_gt += all_metrics[i].n_gt
        n_st += all_metrics[i].n_st
        c += all_metrics[i].c
        g += all_metrics[i].g
        fp += all_metrics[i].fp
        missed += all_metrics[i].missed
        ids += all_metrics[i].mme
        overlap_sum += sum(sum(all_metrics[i].d))
    IDP = IDTP / (IDTP + IDFP) * 100                                # IDP = IDTP / (IDTP + IDFP)
    IDR = IDTP / (IDTP + IDFN) * 100                                # IDR = IDTP / (IDTP + IDFN)
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100                     # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    FAR = fp /  f_gt
    MOTP = (overlap_sum / c) * 100
    MOTAL = (1 - (fp + missed + np.log10(ids + 1)) / g) * 100       # MOTAL = 1 - (# fp + # fn + #log10(ids)) / # gts
    MOTA = (1 - (fp + missed + ids) / g) * 100                      # MOTA = 1 - (# fp + # fn + # ids) / # gts
    recall = c / g * 100                                            # recall = TP / (TP + FN) = # corrected boxes / # gt boxes
    precision = c / (fp + c) * 100                                  # precision = TP / (TP + FP) = # corrected boxes / # det boxes
    metrics = [IDF1, IDP, IDR, recall, precision, FAR, n_gt, MT, PT, ML, fp, missed, ids, FRA, MOTA, MOTP, MOTAL]
    return metrics
    
def evaluate_tracking(sequences, track_dir, gt_dir):
    all_info = []
    for seqname in sequences:
        track_res = os.path.join(track_dir, seqname, 'res.txt')
        gt_file = os.path.join(gt_dir, seqname, 'gt.txt')
        assert os.path.exists(track_res) and os.path.exists(gt_file), 'Either tracking result or groundtruth directory does not exist'

        trackDB = read_txt_to_struct(track_res)
        gtDB = read_txt_to_struct(gt_file)
        
        gtDB, distractor_ids = extract_valid_gt_data(gtDB)
        metrics, extra_info = evaluate_sequence(trackDB, gtDB, distractor_ids)
        print_metrics(seqname + ' Evaluation', metrics)
        all_info.append(extra_info)
    all_metrics = evaluate_bm(all_info)
    print_metrics('Summary Evaluation', all_metrics)

def evaluate_new(stDB, gtDB, distractor_ids):
    
    #trackDB = read_txt_to_struct(results)
    #gtDB = read_txt_to_struct(gt_file)

    #gtDB, distractor_ids = extract_valid_gt_data(gtDB)

    metrics, extra_info, clear_mot_info, ML_PT_MT, M = evaluate_sequence(stDB, gtDB, distractor_ids)

    #print_metrics(' Evaluation', metrics)

    return clear_mot_info, M

@ex.automain
def my_main(_config):

    print(_config)

    ##########################
    # Initialize the modules #
    ##########################
    
    print("[*] Beginning evaluation...")
    output_dir = osp.join(get_output_dir('MOT_analysis'), 'occlusion')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    sequences_raw = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]
    detections = "DPM"
    sequences = ["{}-{}".format(s, detections) for s in sequences_raw]
    
    tracker = ["FRCNN", "DMAN", "HAM_SADF17", "MOTDT17", "EDMT17", "IOU17", "MHT_bLSTM", "FWT_17", "jCC", "MHT_DAM_17"]
    #tracker = ["FRCNN"]
    # "PHD_GSDL17" does not work, error
    #tracker = tracker[-4:]

    for t in tracker:
        print("[*] Evaluating {}".format(t))
        coverage = []
        id_recovered = []
        tr_id_recovered = []
        for s in sequences:
            ########################################
            # Get DPM / GT coverage for each track #
            ########################################

            gt_file = osp.join(cfg.DATA_DIR, "MOT17Labels", "train", s, "gt", "gt.txt")
            det_file = osp.join(cfg.DATA_DIR, "MOT17Labels", "train", s, "det", "det.txt")
            res_file = osp.join(output_dir, t, s+".txt")

            #gtDB = read_txt_to_struct(gt_file)
            #gtDB = gtDB[gtDB[:,7] == 1]

            stDB = read_txt_to_struct(res_file)
            gtDB = read_txt_to_struct(gt_file)

            gtDB, distractor_ids = extract_valid_gt_data(gtDB)

            _, M = evaluate_new(stDB, gtDB, distractor_ids)

            gt_frames = np.unique(gtDB[:, 0])
            st_ids = np.unique(stDB[:, 1])
            gt_ids = np.unique(gtDB[:, 1])
            f_gt = len(gt_frames)
            n_gt = len(gt_ids)
            n_st = len(st_ids)

            gt_inds = [{} for i in range(f_gt)]
            st_inds = [{} for i in range(f_gt)]

            # hash the indices to speed up indexing
            for i in range(gtDB.shape[0]):
                frame = np.where(gt_frames == gtDB[i, 0])[0][0]
                gid = np.where(gt_ids == gtDB[i, 1])[0][0]
                gt_inds[frame][gid] = i

            # Loop thorugh all gt and find gaps (visibility < 0.5)
            visible = [[0 for j in range(f_gt)] for i in range(n_gt)] # format visible[track][frame] = {0,1}
            for gid in range(n_gt):
                for frame in range(f_gt):
                    if gid in gt_inds[frame]:
                        line = gt_inds[frame][gid]
                        vis = gtDB[line, 8]
                        #print(vis, frame, gid)
                        if vis >= 0.5:
                            visible[gid][frame] = 1

            # Find gaps in the tracks
            gt_tracked = {}
            for f,v in enumerate(M):
                for gt in v.keys():
                    if gt not in gt_tracked:
                        gt_tracked[gt] = []
                    gt_tracked[gt].append(f)

            for gid, times in gt_tracked.items():
                times = np.array(times)
                for i in range(len(times)-1):
                    t0 = times[i]
                    t1 = times[i+1]
                    if t1 == t0+1:
                        continue

                    last_non_empty = -1
                    for j in range(t0, -1, -1):
                        if gid in M[j].keys():
                            last_non_empty = j
                            break
                    next_non_empty = -1
                    for j in range(t1, f_gt):
                        if gid in M[j]:
                            next_non_empty = j
                            break

                    if next_non_empty != -1 and last_non_empty != -1:
                        sid0 = M[last_non_empty][gid]
                        sid1 = M[next_non_empty][gid]
                        if sid1 == sid0:
                            tr_id_recovered.append([t1-t0-1, 1])
                        else:
                            tr_id_recovered.append([t1-t0-1, 0])

            """for gid in range(n_gt):
                f0 = -1
                count = 0
                for frame in range(f_gt):
                    if gid in gt_inds[frame]:
                        vis = gtDB[gt_inds[frame][gid], 8]
                        if vis < 0.5 and f0 != -1:
                            count += 1
                        elif vis >= 0.5:
                            if count != 0:
                                print("Gap found {} - {} ({})".format(gid, frame, count))
                                count = 0
                            # set to current frame
                            f0 = frame"""



            # Now iterate through the tracks and check if covered / id kept in comparison to occlusion
            for gid, vis in enumerate(visible):
                f0 = -1
                count = 0
                n_cov = 0
                for frame, v in enumerate(vis):
                    if v == 0 and f0 != -1:
                        count += 1
                        if gid in M[frame].keys():
                            n_cov += 1
                    elif v == 1:
                        # gap ended
                        if count != 0:
                            coverage.append([count, n_cov])

                            last_non_empty = -1
                            for j in range(f0, -1, -1):
                                if gid in M[j].keys():
                                    last_non_empty = j
                                    break
                            next_non_empty = -1
                            for j in range(f0+count+1, f_gt):
                                if gid in M[j]:
                                    next_non_empty = j
                                    break

                            if next_non_empty != -1 and last_non_empty != -1:
                                sid0 = M[last_non_empty][gid]
                                sid1 = M[next_non_empty][gid]
                                if sid1 == sid0:
                                    id_recovered.append([count, 1])
                                else:
                                    id_recovered.append([count, 0])
                            count = 0
                            n_cov = 0
                        # set to current frame
                        f0 = frame

        coverage = np.array(coverage)
        id_recovered = np.array(id_recovered)
        tr_id_recovered = np.array(tr_id_recovered)

        #for c in coverage:
        #    print(c)
        xmax = 50

        # build values for plot
        x_val = np.arange(1,xmax+1)
        y_val = np.zeros(xmax)

        for x in x_val:
            y = np.mean(coverage[coverage[:,0] == x, 1] / coverage[coverage[:,0] == x, 0])
            y_val[x-1] = y

        #plt.plot([0,1], [0,1], 'r-')
        plt.figure()
        plt.scatter(coverage[:,0], coverage[:,1]/coverage[:,0], s=2**2)
        plt.plot(x_val, y_val, 'rx')
        plt.xlabel('gap length')
        plt.xlim((0, xmax))
        plt.ylabel('tracker coverage')
        plt.savefig(osp.join(output_dir, "{}-{}-{}.pdf".format(t, detections, 'GAP_COV')), format='pdf')

        # build values for plot
        x_val = np.arange(1,xmax+1)
        y_val = np.zeros(xmax)

        for x in x_val:
            y = np.mean(id_recovered[id_recovered[:,0] == x, 1])
            y_val[x-1] = y

        plt.figure()
        plt.plot(x_val, y_val, 'rx')
        plt.scatter(id_recovered[:,0], id_recovered[:,1], s=2**2)
        plt.xlabel('gt gap length')
        plt.xlim((0, xmax))
        plt.ylabel('part id recovered')
        plt.savefig(osp.join(output_dir, "{}-{}-{}.pdf".format(t, detections, 'GAP_ID')), format='pdf')
        plt.close()

        # tr id recovered
        x_val = np.arange(1,xmax+1)
        y_val = np.zeros(xmax)

        for x in x_val:
            y = np.mean(tr_id_recovered[tr_id_recovered[:,0] == x, 1])
            y_val[x-1] = y

        plt.figure()
        plt.plot(x_val, y_val, 'rx')
        plt.scatter(tr_id_recovered[:,0], tr_id_recovered[:,1], s=2**2)
        plt.xlabel('track gap length')
        plt.xlim((0, xmax))
        plt.ylabel('part id recovered')
        plt.savefig(osp.join(output_dir, "{}-{}-{}.pdf".format(t, detections, 'GAP_TR_ID')), format='pdf')
        plt.close()
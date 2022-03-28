"""
Given the path to the directory that contains per-frame proposals,
perform frame-level Non-maxima Suppression (NMS)

Using TAO dataset as an example, the input folder structure is presumed to be:

inputdir/
    ArgoVerse/
        video01/
            frame001.npz or frame001.json
            frame002.npz or frame002.json
            ...
        video02/
        ...
    AVA/
    BDD/
    ...

"""

import argparse
import glob
import json
import math
import numpy as np
import os
import torch
import tqdm
import warnings

from pycocotools.mask import encode, decode, iou, toBbox
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms


# =======================================================
# The correctness of `mask-based-NMS` is not guaranteed
# =======================================================

# def compute_iou_for_binary_segmentation(y_argmax, target):
#     I = np.logical_and(y_argmax == 1, target == 1).sum()
#     U = np.logical_or(y_argmax == 1, target == 1).sum()
#     if U == 0:
#         IOU = 1.0
#     else:
#         IOU = float(I) / U
#     return IOU
#
#
# def nms_mask(masks, confidence_score, threshold=0.5):
#     """
#     Args:
#         masks: List, each instance mask is in the form of RLE.
#         confidence_score: List, Confidence score of the masks
#         threshold: float, IoU threshold
#
#     Returns:
#         List, masks and scores that remains
#     """
#     # If no bounding boxes, return empty list
#     if len(masks) == 0:
#         return [], []
#
#     # Confidence scores of bounding boxes
#     score = np.array(confidence_score)
#
#     # Picked bounding boxes
#     picked_masks = []
#     picked_score = []
#
#     # Sort by confidence score of masks
#     order = np.argsort(score)
#
#     remained_masks = masks.copy()  # masks remains to be evaluated
#     remained_scores = confidence_score.copy()  # masks remains to be evaluated
#     last_len = -1
#     while True:
#         # The index of largest confidence score
#         index = order[-1]
#
#         # Pick the mask with largest confidence score
#         picked_masks.append(remained_masks[index])
#         picked_score.append(remained_scores[index])
#         remained_masks.pop(index)
#         remained_scores.pop(index)
#         if not remained_masks:
#             break
#
#         # Compare the IoUs of the rest of the masks with current mask
#         iscrowd_flags = [int(False)] * len(remained_masks)
#         ious = iou([picked_masks[-1]], remained_masks, pyiscrowd=iscrowd_flags)
#         ious = ious.squeeze()
#
#         remained_idx = np.where(ious < threshold)[0]
#         if remained_idx.size == 0:
#             # every mask in the remained_mask is invalid,
#             # because they all overlap with current mask with IoU > threshold
#             break
#         last_len = remained_idx.size
#
#         # Update masks and their corresponding scores remained to be evaluated
#         tmp = [remained_masks[i] for i in remained_idx]
#         remained_masks = tmp.copy()
#         tmp = [remained_scores[i] for i in remained_idx]
#         remained_scores = tmp.copy()
#         # Re-calculate the order
#         score = np.array(remained_scores)
#         order = np.argsort(score)
#
#     return picked_masks, picked_score


def process_one_frame(seq: str, scoring: str, keys, iou_thres: float, outpath: str, datatype: str, categorywise: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load proposals
    if datatype == ".npz":
        npz_file = np.load(seq, allow_pickle=True)
        proposals = npz_file['arr_0'].tolist()
    elif datatype == ".json":
        with open(seq, 'r') as f:
            proposals = json.load(f)
    else:
        raise Exception("The given data type for proposals is not allowed", datatype)


    props_for_nms = dict()
    if not categorywise:
        props_for_nms["cat_agnostic"] = dict()
        for k in keys:
            props_for_nms["cat_agnostic"][k + 's'] = list()

    for prop in proposals:
        if categorywise:
            cat_id = prop['category_id']
        else:
            cat_id = "cat_agnostic"

        # You can add more scoring-strategies below by simply extending the "elif".
        if scoring == "one_minus_bg_score":
            curr_score = 1 - prop['bg_score']
        elif scoring == "bg_obj_sum":
            curr_score = (1000 * prop["objectness"] + prop["bg_score"]) / 2
        elif scoring == "bg_obj_prod":
            curr_score = math.sqrt(1000 * prop["objectness"] * prop["bg_score"])
        else:
            curr_score = prop[scoring]

        if cat_id not in props_for_nms.keys():
            props_for_nms[cat_id] = dict()
            for k in keys:
                props_for_nms[cat_id][k + 's'] = list()

        for k in keys:
            if k == scoring:
                props_for_nms[cat_id][k + 's'].append(curr_score)
            else:
                props_for_nms[cat_id][k + 's'].append(prop[k])


    output = list()
    post_nms = dict()
    for k in keys:
        post_nms[k + 's'] = list()

    if args.nms_criterion == 'bbox':
        """
        torchvision.ops.nms
            Args:
                boxes: Tensor[N,4] - boxes in (x1,y1,x2,y2) format
                scores: Tensor[N] 
                iou_threshold: float
            Return:
                keep: int64 - tensor with the indices of the elements that have been kept by NMS, 
                              sorted in decreasing order of scores
        """
        for cat_id, data in props_for_nms.items():
            boxes_tensor = torch.Tensor(props_for_nms[cat_id]['bboxs']).to(device)
            scores_tensor = torch.Tensor(props_for_nms[cat_id][scoring + 's']).to(device)

            keep = nms(boxes=boxes_tensor, scores=scores_tensor, iou_threshold=0.5)
            post_nms['bboxs'] = boxes_tensor[keep].cpu().tolist()
            post_nms[scoring + 's'] = scores_tensor[keep].cpu().tolist()

            keep = keep.cpu().tolist()
            for k in keys:
                if k == 'bbox' or k == scoring:
                    continue
                post_nms[k + 's'] = [props_for_nms[cat_id][k + 's'][i] for i in keep]

    elif args.nms_criterion == 'instance_mask':
        # props_nms_mask, scores_nms_mask = nms_mask(props_for_nms['props'], props_for_nms['scores'], iou_thres)
        pass  # this part is not implemented
    else:
        raise Exception(args.nms_criterion, "invalid. Please choose from `bbox` or `mask`")

    if args.nms_criterion == 'bbox':
        for i in range(len(post_nms['bboxs'])):
            curr_prop = dict()
            for k in keys:
                curr_prop[k] = post_nms[k + 's'][i]
            output.append(curr_prop)

        # for box, mask, embed, score in zip(props_nms_box, props_nms_mask, props_nms_embed, scores_nms):
        #     output.append({'bbox': box, 'instance_mask': mask, 'embeddings': embed, scoring: score})
    elif args.nms_criterion == 'instance_mask':
        pass
        # for prop, score in zip(props_nms, scores_nms):
        #     box = toBbox(prop).tolist()  # in [xc, yc, w, h]
        #     # convert [xc, yc, w, h] to [x1, y1, x2, y2]
        #     bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        #     output.append({'bbox': bbox, args.nms_criterion: prop, scoring: score})

    # Store proposals after NMS
    outdir = "/".join(outpath.split("/")[:-1])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if datatype == ".npz":
        np.savez_compressed(outpath, output)
    elif datatype == ".json":
        with open(outpath, 'w') as f:
            json.dump(output, f)


def process_all_folders(root_dir: str, scoring: str, keys, iou_thres: float, outdir: str, datatype: str):
    datasrc_list = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, '*')))]
    if args.datasrcs:
        datasrc_list = args.datasrcs

    print(">>>>>>>>> Perform NMS for the following datasets: {}".format(datasrc_list))
    for datasrc in datasrc_list:
        print("Processing", datasrc)
        video_names = sorted([fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, datasrc, '*')))])

        for idx, video_name in enumerate(tqdm.tqdm(video_names)):
            all_seq = glob.glob(os.path.join(root_dir, datasrc, video_name, "*.npz"))
            for seq in all_seq:
                json_name = seq.split("/")[-1]
                outpath = os.path.join(outdir + "_" + scoring, datasrc, video_name, json_name)
                process_one_frame(seq, scoring, keys, iou_thres, outpath, datatype, args.categorywise)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorings", required=True, nargs="+",
                        help="different scorings to use during NMS, implemented scorings:"
                             "objectness" "bg_score" "score" "one_minus_bg_score" "bg_obj_sum" "bg_obj_prod"
                             "it's possible to implement more complicated scorings.")

    parser.add_argument('--iou_thres', default=0.5, type=float, help='IoU threshold used in NMS')
    parser.add_argument('--nms_criterion', default='bbox', type=str, help='NMS based on bbox or mask')
    parser.add_argument('--categorywise', action='store_true',
                        help='Only perform NMS among the same category.'
                             'If not enabled, perform NMS globally on every bboxes, ignoring their categories')
    parser.add_argument('--inputdir', required=True, type=str,
                        help='input directory containing proposals from all sources of datasets')
    parser.add_argument('--outdir', required=True, type=str, help='output directory containing proposals after NMS')
    parser.add_argument('--datasrcs', nargs='+', type=str, help='sources of datasets to process')
    parser.add_argument('--datatype', type=str, default=".npz", choices=[".npz", ".json"], help="the data format of input")
    args = parser.parse_args()

    for scoring in args.scorings:

        # The information you choose to store for each proposal.
        # They must already exist for all the proposals before performing NMS.
        # bbox and scoring are necessary for NMS.
        keys = ['bbox', scoring, 'category_id', 'instance_mask', 'embeddings']

        print(">>>>>>>>> NMS using {}".format(scoring))
        process_all_folders(args.inputdir, scoring, keys, args.iou_thres, args.outdir, args.datatype)
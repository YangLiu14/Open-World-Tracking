import glob
import numpy as np
import os
import torch

from PIL import Image
from pycocotools.mask import encode, decode, toBbox
from pycocotools.mask import iou as mask_iou
from sklearn.metrics.pairwise import cosine_similarity

from utils import open_flow_png_file, warp_flow, bbox_iou, compute_giou, readFlow, calculate_flow


__all__ = ('similarity_mix',
           )


def similarity_mix(prop_L, props_R, frameL, frameR, opt_flow_dir, use_frames_in_between=False):
    """
    Current hybrid mode is in "direct" setting. The combined score is calculated as:
    `0.5 * Optical_Flow (bbox_iou) + 0.5 * MRCNN (reid)`
    """
    # Extract information from prop_L
    mask_L = decode(prop_L['instance_mask'])
    embed_L = prop_L['embeddings'].reshape(1, -1)

    flow = readFlow(os.path.join(opt_flow_dir, frameL + '.flo'))
    warped_mask = warp_flow(mask_L, flow)  # warp flow to next frame

    # Match warped-mask with the proposals in last frame
    # masks_R = [prop['instance_mask'] for prop in props_R]
    bboxs_R = [prop['bbox'] for prop in props_R]
    warped_mask = encode(np.array(warped_mask[:, :, np.newaxis], order='F'))[0]
    warped_mask['counts'] = warped_mask['counts'].decode(encoding="utf-8")

    x, y, w, h = toBbox(warped_mask).tolist()
    OF_iou_scores = np.array([bbox_iou([x, y, x + w, y + h], box) for box in bboxs_R])

    # MRCNN embeddings
    cosine_scores = np.array([cosine_similarity(embed_L, prop_R['embeddings'].reshape(1, -1)) for prop_R in props_R])
    cosine_scores = cosine_scores.squeeze()
    combined_scores = 0.5 * OF_iou_scores + 0.5 * cosine_scores

    return combined_scores

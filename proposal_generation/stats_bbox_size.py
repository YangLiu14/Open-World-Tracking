import argparse
import glob
import json
import numpy as np
import os
import tqdm

from collections import Counter


# =======================================================
# Global variables
# =======================================================
"""
known_tao_ids: set of tao ids that can be mapped exactly to coco ids.
neighbor_classes: tao classes that are similar to coco_classes.
unknown_tao_ids: all_tao_ids that exclude known_tao_ids and neighbor_classes..
"""

# all_ids = set([i for i in range(1, 1231)])
all_ids = set([i for i in range(1, 1642)])

# Category IDs in TAO that are known (appeared in COCO)
with open("./datasets/coco_id2tao_id.json") as f:
    coco_id2tao_id = json.load(f)
known_tao_ids = set([v for k, v in coco_id2tao_id.items()])
# Category IDs in TAO that are unknown (comparing to COCO)
unknown_tao_ids = all_ids.difference(known_tao_ids)

# neighbor classes
with open("./datasets/neighbor_classes.json") as f:
    coco2neighbor_classes = json.load(f)
# Gather tao_ids that can be categorized in the neighbor_classes
neighbor_classes = set()
for coco_id, neighbor_ids in coco2neighbor_classes.items():
    neighbor_classes = neighbor_classes.union(set(neighbor_ids))

# Exclude neighbor classes from unknown_tao_ids
unknown_tao_ids = unknown_tao_ids.difference(neighbor_classes)

# =======================================================
# =======================================================


def map_video_id2size(gt_dict):
    """
    Map video_id -> image-size
    """
    v_id2size = dict()
    for v in gt_dict['videos']:
      v_id2size[v["id"]] = {"width": v["width"], "height": v["height"]}

    return v_id2size


def anaylse_bbox_size(gt_path: str):
    # The bbox annotations are in [x, y, w, h]
    with open(gt_path, 'r') as f:
        gt_dict = json.load(f)

    v_id2size = map_video_id2size(gt_dict)

    ratio2cnt = Counter()
    for ann in gt_dict['annotations']:
        x, y, w, h = ann['bbox']
        img_w, img_h = v_id2size[ann["video_id"]]["width"], v_id2size[ann["video_id"]]["height"]
        ratio = (w * h) / (img_w * img_h)
        ratio = "{:.2f}".format(ratio)
        ratio2cnt[ratio] += 1

    # Sort by ratio
    ratio_list = list()
    for ratio, cnt in ratio2cnt.items():
        ratio_list.append((ratio, cnt))
    ratio_list.sort(key=lambda r: r[0])

    # Split into large, medium, small
    large, medium, small = 0, 0, 0
    for itm in ratio_list:
        ratio = float(itm[0])
        box_cnt = int(itm[1])
        if ratio >= 0.3:
            large += box_cnt
        elif 0.03 <= ratio < 0.3:
            medium += box_cnt
        else:
            small += box_cnt

    print("Large", large)
    print("Medium", medium)
    print("Small", small)


# ---------------------------------------------------------------
def load_gt(gt_path, bounds, exclude_classes=(), ignored_sequences=()):
    with open(gt_path, 'r') as f:
        gt_json = json.load(f)

    videos = gt_json['videos']
    annotations = gt_json['annotations']
    tracks = gt_json['tracks']
    images = gt_json['images']
    info = gt_json['info']
    categories = gt_json['categories']

    v_id2size = map_video_id2size(gt_json)

    gt = {}
    imgID2fname = dict()
    for img in images:
        imgID2fname[img['id']] = img['file_name']

    nbox_large, nbox_medium, nbox_small = 0, 0, 0

    for ann in annotations:
        if ann["category_id"] in exclude_classes:
            continue

        img_id = ann['image_id']
        img_w, img_h = v_id2size[ann["video_id"]]["width"], v_id2size[ann["video_id"]]["height"]
        fname = imgID2fname[img_id]
        fname = fname.replace("jpg", "npz")

        # ignore certain data souces
        src_name = fname.split("/")[1]
        if src_name in ignored_sequences:
            continue

        x, y, w, h = ann['bbox']
        # convert [x, y, w, h] to [x1, y1, x2, y2]
        bbox = [x, y, x + w, y + h]
        # Determine box type (large, medium, small)
        ratio = (w * h) / (img_w * img_h)
        ratio = float("{:.2f}".format(ratio))
        box_type = None
        if ratio >= bounds[1]:
            box_type = 'large'
            nbox_large += 1
        elif bounds[0] <= ratio < bounds[1]:
            box_type = 'medium'
            nbox_medium += 1
        else:
            box_type = 'small'
            nbox_small += 1

        if fname not in gt.keys():
            gt[fname] = {'bboxes': [], 'box_types': [], 'track_ids': []}
        gt[fname]['bboxes'].append(bbox)
        gt[fname]['box_types'].append(box_type)
        gt[fname]['track_ids'].append(ann['track_id'])

    n_boxes = sum([len(x['bboxes']) for x in gt.values()], 0)
    assert n_boxes == nbox_large + nbox_medium + nbox_small
    print("number of gt boxes", n_boxes)
    nbox = {"large": nbox_large,
            "medium": nbox_medium,
            "small": nbox_small}
    print(nbox)

    return gt, n_boxes, nbox


def calculate_ious(bboxes1, bboxes2):
    """
    :param bboxes1: Kx4 matrix, assume layout (x0, y0, x1, y1)
    :param bboxes2: Nx$ matrix, assume layout (x0, y0, x1, y1)
    :return: KxN matrix of IoUs
    """
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    I = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    U = area1[:, np.newaxis] + area2[np.newaxis, :] - I
    assert (U > 0).all()
    IOUs = I / U
    assert (IOUs >= 0).all()
    assert (IOUs <= 1).all()
    return IOUs


def evaluate_proposals(gt_per_frame, props, iou_threshold):
    prop_boxes = np.array([prop["bbox"] for prop in props])
    gt_boxes = np.array(gt_per_frame['bboxes'])
    ious = calculate_ious(gt_boxes, prop_boxes)
    max_ious = np.max(ious, 1)
    assert len(max_ious) == len(gt_per_frame['box_types'])

    large, medium, small = 0, 0, 0
    for i in range(len(max_ious)):
        type = gt_per_frame['box_types'][i]
        curr_iou = max_ious[i]
        add = 1 if curr_iou >= iou_threshold else 0
        if type == 'large':
            large += add
        elif type == 'medium':
            medium += add
        elif type == 'small':
            small += add

    return large, medium, small


def recall_based_on_bbox_size(props_dir: str, gt_path: str, outdir: str, iou_threshold: float, bounds, split):
    recall_L, recall_M, recall_S = 0, 0, 0

    if split == "known":
        print("evaluating coco 78 classes")
        exclude_classes = tuple(unknown_tao_ids.union(neighbor_classes))
        gt, total_boxes, nbox = load_gt(gt_path, bounds, exclude_classes=exclude_classes)

    elif split == "neighbor":
        print("evaluating neighbor classes")
        exclude_classes = tuple(known_tao_ids.union(unknown_tao_ids))
        gt, total_boxes, nbox = load_gt(gt_path, bounds, exclude_classes=exclude_classes)

    elif split == "unknown":
        print("evaluating unknown")
        exclude_classes = tuple(known_tao_ids.union(neighbor_classes))
        gt, total_boxes, nbox = load_gt(gt_path, bounds, exclude_classes=exclude_classes)

    """ gt (Dict):
    {frame_name: {"bboxes: List[[x1,y1,x2,y2]], "track_ids": List[int]}}
    """

    datasrcs = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(props_dir, '*')))]
    print(datasrcs)
    for datasrc in datasrcs:
        print("Processing", datasrc)
        videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(props_dir, datasrc, '*')))]
        for idx, video in enumerate(tqdm.tqdm(videos)):
            v_dir = os.path.join(props_dir, datasrc, video)
            # List all files in the current folder
            files = sorted(glob.glob(v_dir + '/*' + '.npz'))
            for frame_path in files:
                key = 'val/' + '/'.join(frame_path.split('/')[-3:])
                if key not in gt.keys():
                    continue
                props = np.load(frame_path, allow_pickle=True)['arr_0'].tolist()
                L, M, S = evaluate_proposals(gt[key], props, iou_threshold)

                recall_L += L
                recall_M += M
                recall_S += S
        print("-------------- Current Results --------------")
        print("Large: {}/{} = {}".format(recall_L, nbox["large"], recall_L / nbox["large"]))
        print("Medium: {}/{} = {}".format(recall_M, nbox["medium"], recall_M / nbox["medium"]))
        print("Small: {}/{} = {}".format(recall_S, nbox["small"], recall_S / nbox["small"]))

    print(nbox)
    print("-------------- Final Results --------------")
    print("Large: {}/{} = {}".format(recall_L, nbox["large"], recall_L / nbox["large"]))
    print("Medium: {}/{} = {}".format(recall_M, nbox["medium"], recall_M / nbox["medium"]))
    print("Small: {}/{} = {}".format(recall_S, nbox["small"], recall_S / nbox["small"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script for Proposal Similarity')
    parser.add_argument('--props_dir', required=True, type=str,
                        help='Root directory where proposals are stored. Proposals are in npz file.')
    parser.add_argument('--gt_path', required=True, type=str, help='File path to GT annotation file.')
    parser.add_argument('--iou_threshold', default=0.5, type=float,
                        help='When proposal and gt-object have IoU above this IoU threshold, the gt is recalled')
    parser.add_argument('--outdir', type=str, help='Output directory of intermediate results')
    parser.add_argument('--split', type=str, choices=['known', 'neighbor', 'unknown'])
    parser.add_argument('--bound_large', type=float, default=0.3,
                        help='box_ratio = box_size / image_size. if ratio >= bound_large, then the box is large')
    parser.add_argument('--bound_small', type=float, default=0.03,
                        help='box_ratio = box_size / image_size. if ratio < bound_small, then the box is small')

    args = parser.parse_args()
    # anaylse_bbox_size(args.gt_path)

    recall_based_on_bbox_size(args.props_dir, args.gt_path, args.outdir, args.iou_threshold,
                              (args.bound_small, args.bound_large), args.split)

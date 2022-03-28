import argparse
import csv
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import tqdm


from collections import Counter
from sklearn import metrics


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
with open("./datasets/neighbor_classes_old.json") as f:
    coco2neighbor_classes = json.load(f)
# Gather tao_ids that can be categorized in the neighbor_classes
neighbor_classes = set()
for coco_id, neighbor_ids in coco2neighbor_classes.items():
    neighbor_classes = neighbor_classes.union(set(neighbor_ids))

# Exclude neighbor classes from unknown_tao_ids
unknown_tao_ids = unknown_tao_ids.difference(neighbor_classes)
# =======================================================


# =======================================================
# plot functions
# =======================================================
def plot_from_csv(csv_folder: str, outdir: str, postNMS: bool, n_max_props: int):

    def scoring2suffix(scoring: str):
        if scoring == "score":
            return "score"
        elif scoring == "bgScore":
            return "bg_score"
        elif scoring == "1-bgScore":
             return "one_minus_bg_score"
        elif scoring == "objectness":
            return "objectness"
        elif scoring == "bg+objectness":
            return "bg_obj_sum"
        elif scoring == "bg*objectness":
            return "bg_obj_prod"
        else:
            return ''

    scoring_list = ["score", "bgScore", "1-bgScore", "objectness", "bg+objectness", "bg*objectness"]
    export_dict = {"score": {}, "bgScore": {}, "1-bgScore": {},
                   "objectness": {}, "bg+objectness": {}, "bg*objectness": {}}

    print("Read in csv files")
    for split in ["known", "neighbor", "unknown"]:
        for scoring in tqdm.tqdm(scoring_list):
            # if postNMS:
            csv_path = os.path.join(csv_folder, split + '_' + scoring2suffix(scoring) + ".csv")
            # else:
            #     csv_path = os.path.join(csv_folder, split + ".csv")
            y = list()
            with open(csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                for row in reader:
                    y.append(float(row[0]))
            export_dict[scoring]["data"] = y

        x_vals = range(n_max_props + 1)
        plot_title = split + " classes"
        outdir = os.path.join(csv_folder, "final")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        make_plot(outdir, split, export_dict, plot_title, x_vals, linewidth=5, n_max_props=n_max_props)


def title_to_filename(plot_title):
    filtered_title = re.sub("[\(\[].*?[\)\]]", "", plot_title)  # Remove the content within the brackets
    filtered_title = filtered_title.replace("_", "").replace(" ", "").replace(",", "_")
    return filtered_title


def make_plot(output_dir, split, export_dict, plot_title, x_vals, linewidth=5, n_max_props=1000):
    plt.figure()

    itm = export_dict.items()
    itm = sorted(itm, reverse=True)
    for idx, item in enumerate(itm):
        # Compute Area Under Curve
        x = x_vals[0:n_max_props]
        y = item[1]['data'][0:n_max_props]
        auc = round(metrics.auc(x, y), 2)

        auc = auc / n_max_props
        auc = round(auc, 2)

        curr_scoring = "score"
        if item[0].replace('.', '') == "objectness":
            curr_scoring = "obj."
        elif item[0].replace('.', '') == "bgScore":
            curr_scoring = "bg"
        elif item[0].replace('.', '') == "bg+objectness":
            curr_scoring = "bg+obj."
        elif item[0].replace('.', '') == "bg*objectness":
            curr_scoring = 'bg*obj.'
        elif item[0].replace('.', '') == "1-bgScore":
            curr_scoring = "1-bg"

        # curve_label = item[0].replace('.', '') + ': '  'AUC=' + str(auc)
        curve_label = curr_scoring + ' (' + str(auc) + ')'
        plt.plot(x_vals[0:n_max_props], item[1]['data'][0:n_max_props], label=curve_label, linewidth=linewidth)

    ax = plt.gca()
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    if n_max_props == 1000:
        ax.set_xticks(np.asarray([10, 200, 500, 700, 1000]))
    else:
        ax.set_xticks(np.asarray([1, 20, 50, 70, 100, 150, 200]))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(f"$\#$ proposals ({split})", fontsize=25)
    plt.ylabel("Proposal Recall", fontsize=25)
    ax.set_ylim([0.0, 1.0])
    plt.legend(prop={"size": 16}, handlelength=1.0, labelspacing=0.2, borderaxespad=0.2, loc='lower right')
    plt.grid()
    # plt.title(plot_title)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, title_to_filename(plot_title) + ".pdf"), bbox_inches='tight')


# =======================================================
# Eval functions
# =======================================================
def load_gt(gt_path, exclude_classes=(), ignored_datasrcs=()):
    """
    Args:
        gt_path (str): file path to ground truth json.
        exclude_classes (tuple): gt object with class ids in this tuple will be not be loaded.
        ignored_sequences:

    Returns:
        gt (Dict): {frame_name: {"bboxes: List[[x1,y1,x2,y2]], "track_ids": List[int]}}
    """
    with open(gt_path, 'r') as f:
        gt_json = json.load(f)

    annotations = gt_json['annotations']
    images = gt_json['images']

    gt = {}
    imgID2fname = dict()
    for img in images:
        imgID2fname[img['id']] = img['file_name']

    for ann in annotations:
        if ann["category_id"] in exclude_classes:
            continue

        img_id = ann['image_id']
        fname = imgID2fname[img_id]
        fname = fname.replace("jpg", "npz")

        # ignore certain data sources
        src_name = fname.split("/")[1]
        if src_name in ignored_datasrcs:
            continue

        x, y, w, h = ann['bbox']
        # convert [x, y, w, h] to [x1, y1, x2, y2]
        bbox = [x, y, x + w, y + h]

        if fname not in gt.keys():
            gt[fname] = {'bboxes': [], 'track_ids': []}
        gt[fname]['bboxes'].append(bbox)
        gt[fname]['track_ids'].append(ann['track_id'])

    n_boxes = sum([len(x['bboxes']) for x in gt.values()], 0)

    print("number of gt boxes", n_boxes)
    return gt, n_boxes


def load_proposals(props_dir: str, scoring: str, datasplit: str):
    f"""
    Load proposals from all data-sources that are present in `props_dir`
    Args:
        props_dir (str): Directory containing proposals, potentially from all data-sources (ArgoVerse, BDD, etc.)
        scoring (str): Proposal per each frame will be sorted by this scoring, high score --> low score.
        datasplit (str): `train` or `val` 

    Returns:
        proposals (dict): key = full frame name (contains datasrc and video name), 
                          value = list of bbox (x1, y1, x2, y2), sorted by the given `scoring`
    """
    print("Loading all proposals")
    proposals = {}
    datasrcs = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(props_dir, '*')))]
    print(datasrcs)
    for datasrc in datasrcs:
        print("Loading", datasrc)
        videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(props_dir, datasrc, '*')))]
        for idx, video in enumerate(tqdm.tqdm(videos)):
            v_dir = os.path.join(props_dir, datasrc, video)
            files = sorted(glob.glob(v_dir + '/*' + '.npz'))
            for frame_path in files:
                key = datasplit + '/' + '/'.join(frame_path.split('/')[-3:])
                props = np.load(frame_path, allow_pickle=True)['arr_0'].tolist()
                # Sort proposals according to different scoring
                if scoring in props[0].keys():
                    props.sort(key=lambda p: p[scoring], reverse=True)
                else:
                    if scoring == "one_minus_bg_score":
                        props.sort(key=lambda p: 1 - p["bg_score"], reverse=True)
                    elif scoring == "bg_obj_sum":
                        props.sort(key=lambda p: (1000 * p["objectness"] + p["bg_score"]) / 2, reverse=True)
                    elif scoring == "bg_obj_prod":
                        props.sort(key=lambda p: math.sqrt(1000 * p["objectness"] * p["bg_score"]), reverse=True)

                bboxes = [prop["bbox"] for prop in props]  # [x1, y1, x2, y2]
                proposals[key] = bboxes

    return proposals


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


def evaluate_proposals(all_gt, all_props, iou_threshold, n_max_proposals=1000):
    all_ious = []  # ious for all frames
    all_track_ids = []
    for frame_name, gt in all_gt.items():
        if not all_props.get(frame_name, None):
            continue
        gt_bboxes = np.array(gt['bboxes'])
        track_ids = np.array(gt['track_ids'])
        prop_bboxes = np.array(all_props[frame_name])

        ious = calculate_ious(gt_bboxes, prop_bboxes)
        # pad to n_max_proposals
        ious_padded = np.zeros((ious.shape[0], n_max_proposals))
        ious_padded[:, :ious.shape[1]] = ious[:, :n_max_proposals]
        all_ious.append(ious_padded)

    all_ious = np.concatenate(all_ious)

    iou_curve = [0.0 if n_max == 0 else (all_ious[:, :n_max].max(axis=1) > iou_threshold).mean() for n_max in
                 range(0, n_max_proposals + 1)]

    return iou_curve


def recall_eval(props_dir: str, gt_path: str, outdir: str, iou_threshold: float,
                n_max_props: int, scoring: str, datasplit='val'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    """ all_proposals (Dict):
    { frame_name: List[[x1,y1,x2,y2]] }
    """
    all_proposals = load_proposals(props_dir, scoring, datasplit)

    """ gt (Dict):
    {frame_name: {"bboxes: List[[x1,y1,x2,y2]], "track_ids": List[int]}}
    """
    # --------- Known ----------
    print("evaluating coco 78 classes")
    exclude_classes = tuple(unknown_tao_ids.union(neighbor_classes))
    gt, total_boxes = load_gt(gt_path, exclude_classes=exclude_classes)
    iou_curve_known = evaluate_proposals(gt, all_proposals, iou_threshold, n_max_proposals=n_max_props)
    print("total_boxes:", total_boxes)
    print("iou_curve")
    print(iou_curve_known)
    try:
        np.savetxt(os.path.join(outdir, 'known_{}.csv'.format(scoring)), np.array(iou_curve_known),
                   delimiter=',', fmt='%1.4f')
    except:
        print("unable to save csv")

    # -------- Neighbor ---------
    print("evaluating neighbor classes")
    exclude_classes = tuple(known_tao_ids.union(unknown_tao_ids))
    gt, total_boxes = load_gt(gt_path, exclude_classes=exclude_classes)
    iou_curve_neighbor = evaluate_proposals(gt, all_proposals, iou_threshold, n_max_proposals=n_max_props)
    print("total_boxes:", total_boxes)
    print("iou_curve")
    print(iou_curve_neighbor)
    try:
        np.savetxt(os.path.join(outdir, 'neighbor_{}.csv'.format(scoring)), np.array(iou_curve_neighbor),
                   delimiter=',', fmt='%1.4f')
    except:
        print("unable to save csv")

    # -------- Unknown -----------
    print("evaluating unknown")
    exclude_classes = tuple(known_tao_ids.union(neighbor_classes))
    gt, total_boxes = load_gt(gt_path, exclude_classes=exclude_classes)
    iou_curve_unknown = evaluate_proposals(gt, all_proposals, iou_threshold, n_max_proposals=n_max_props)
    print("total_boxes:", total_boxes)
    print("iou_curve")
    print(iou_curve_unknown)
    try:
        np.savetxt(os.path.join(outdir, 'unknown_{}.csv'.format(scoring)), np.array(iou_curve_unknown),
                   delimiter=',', fmt='%1.4f')
    except:
        print("unable to save csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script for Proposal Similarity')
    parser.add_argument('--outdir', type=str, help='Plots output dir.')
    parser.add_argument('--props_dir', type=str, help='Base directory of the proposals to be evaluated.')
    parser.add_argument('--labels', type=str, default='',
                        help='Specify dir containing the labels. '
                             'In TAO dataset, the ground truths labels are store in a single `xxx.json` file.')
    parser.add_argument('--score_func', type=str, help='Sorting criterion to use. Choose from' +
                                                       '[score, bg_score, 1-bg_score, obj, bg+obj, bg*obj]')
    parser.add_argument('--iou_threshold', default=0.5, type=float,
                        help='When proposal and gt-object have IoU above this IoU threshold, the gt is successfully recalled')
    parser.add_argument('--postNMS', action='store_true', help='processing postNMS proposals.')
    parser.add_argument('--datasplit', default='val', type=str, help='evalution dataset from [train, val, test]')
    parser.add_argument('--n_max_props', type=int, help='Max number of proposals allowed to use to recall GT-objects.')
    args = parser.parse_args()

    scoring_names = ["bg_obj_prod", "bg_obj_sum", "bg_score",
                     "objectness", "one_minus_bg_score", "score"]
    # # old naming conventions
    # scoring_names = ["bg_rpn_product", "bg_rpn_sum", "bg_score",
    #                  "objectness", "one_minus_bg_score", "score"]

    if args.props_dir:
        if args.postNMS:
            # Using proposals (per frame) after NMS to recall GT-objects (per frame).
            all_props_dirs = [args.props_dir + '/' + '_' + sd for sd in scoring_names]
            for idx, props_dir in enumerate(all_props_dirs):
                print("Processing", props_dir)
                recall_eval(props_dir, args.labels, args.outdir, args.iou_threshold,
                            args.n_max_props, scoring=scoring_names[idx], datasplit=args.datasplit)
        else:
            # Using all the proposals (per frame) to recall GT-objects (per frame).
            props_dir = args.props_dir
            for idx, scoring in enumerate(scoring_names):
                print("Processing", scoring)
                recall_eval(props_dir, args.labels, args.outdir, args.iou_threshold,
                            args.n_max_props, scoring=scoring, datasplit=args.datasplit)

    # If previous evaluation are completed and stored as CSV files, leave args.props_dir empty and
    # use the function below to directly ouput plots.
    plot_from_csv(csv_folder=args.outdir, outdir=args.outdir, postNMS=args.postNMS, n_max_props=args.n_max_props)


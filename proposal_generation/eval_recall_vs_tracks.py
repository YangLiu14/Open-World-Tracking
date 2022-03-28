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
def plot_from_csv(csv_folder: str, outdir: str, postNMS):

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

    if postNMS:
        scoring_list = ["score", "bgScore", "1-bgScore", "objectness", "bg+objectness", "bg*objectness"]
        export_dict = {"score": {}, "bgScore": {}, "1-bgScore": {},
                       "objectness": {}, "bg+objectness": {}, "bg*objectness": {}}
    else:
        scoring_list = [""]
        export_dict = {"": {}}

    print("Read in csv files")
    for split in ["known", "neighbor", "unknown"]:
        for scoring in tqdm.tqdm(scoring_list):
            if postNMS:
                csv_path = os.path.join(csv_folder, split + '_' + scoring2suffix(scoring) + ".csv")
            else:
                csv_path = os.path.join(csv_folder, split + ".csv")
            y = list()
            with open(csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                for row in reader:
                    y.append(float(row[0]))
            export_dict[scoring]["data"] = y

        x_vals = range(102)
        plot_title = split + " classes"
        outdir = os.path.join(csv_folder, "final")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        make_plot(outdir, export_dict, plot_title, x_vals, linewidth=5)


def title_to_filename(plot_title):
    filtered_title = re.sub("[\(\[].*?[\)\]]", "", plot_title)  # Remove the content within the brackets
    filtered_title = filtered_title.replace("_", "").replace(" ", "").replace(",", "_")
    return filtered_title


def make_plot(output_dir, export_dict, plot_title, x_vals, linewidth=5):
    plt.figure()

    itm = export_dict.items()
    itm = sorted(itm, reverse=True)
    for idx, item in enumerate(itm):
        # Compute Area Under Curve
        x = x_vals[0:100]
        y = item[1]['data'][0:100]
        auc = round(metrics.auc(x, y), 2)
        curve_label = item[0].replace('.', '') + ': '  'AUC=' + str(auc)
        # curve_label = 'AUC=' + str(auc)
        plt.plot(x_vals[0:100], item[1]['data'][0:100], label=curve_label, linewidth=linewidth)

    ax = plt.gca()
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xticks(np.asarray([1, 10, 20, 30, 50, 70, 90, 100]))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("% detected")
    plt.ylabel("Track Recall")
    ax.set_ylim([0.0, 1.0])
    # plt.legend(prop={"size": 25})
    plt.legend(prop={"size": 25}, handlelength=1.0, labelspacing=0.2, borderaxespad=0.2, loc='lower right')
    plt.grid()
    plt.title(plot_title)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, title_to_filename(plot_title) + ".pdf"), bbox_inches='tight')

# =======================================================
# Eval functions
# =======================================================
def score_func(prop):
    if args.score_func == "score":
        return prop["score"]
    if args.score_func == "bgScore":
        return prop["bg_score"]
    if args.score_func == "1-bgScore":
        if args.postNMS:
            return prop['one_minus_bg_score']
        else:
            return 1 - prop["bg_score"]
    if args.score_func == "objectness":
        return prop["objectness"]
    if args.score_func == "bg+objectness":
        if args.postNMS:
            return prop['bg_obj_sum']
        else:
            return (1000 * prop["objectness"] + prop["bg_score"]) / 2
    if args.score_func == "bg*objectness":
        if args.postNMS:
            return prop['bg_obj_prod']
        else:
            return math.sqrt(1000 * prop["objectness"] * prop["bg_score"])


def load_gt(gt_path, exclude_classes=(), ignored_sequences=()):
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

        # ignore certain data souces
        src_name = fname.split("/")[1]
        if src_name in ignored_sequences:
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


def load_proposals(props_dir, datasplit, ignored_sequences=(), score_fnc=score_func):
    print("Loading all proposals")
    proposals = {}
    datasrcs = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(props_dir, '*')))]
    print(datasrcs)
    for datasrc in datasrcs:
        print("Loading", datasrc)
        videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(props_dir, datasrc, '*')))]
        for idx, video in enumerate(tqdm.tqdm(videos)):
            v_dir = os.path.join(props_dir, datasrc, video)
            # List all files in the current folder
            files = sorted(glob.glob(v_dir + '/*' + '.npz'))
            for frame_path in files:
                key = datasplit + '/' + '/'.join(frame_path.split('/')[-3:])
                props = np.load(frame_path, allow_pickle=True)['arr_0'].tolist()
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
        gt_bboxes = np.array(gt['bboxes'])
        track_ids = np.array(gt['track_ids'])
        prop_bboxes = np.array(all_props[frame_name])
        ious = calculate_ious(gt_bboxes, prop_bboxes)
        # pad to n_max_proposals
        ious_padded = np.zeros((ious.shape[0], n_max_proposals))
        ious_padded[:, :ious.shape[1]] = ious[:, :n_max_proposals]
        all_ious.append(ious_padded)
        all_track_ids.append(track_ids)

    all_ious = np.concatenate(all_ious)
    all_track_ids = np.concatenate(all_track_ids)

    # Calculate the track-length of each track id
    gt_track_len = Counter()
    for t_id in all_track_ids:
        gt_track_len[t_id] += 1

    iou_curve = list()
    # N here means: we only consider a track is successfully recalled,
    # when there are at least % of objects in this track are detected.
    for N in range(101):
        mask = (all_ious[:, :1000].max(axis=1) > iou_threshold)
        recalled_track_ids = all_track_ids[mask]

        # Count how many times one track_id is being recalled
        recalled_track_ids = recalled_track_ids.tolist()
        track_id2cnt = Counter()
        for t_id in recalled_track_ids:
            track_id2cnt[t_id] += 1

        recalled_track_ids = list()
        for t_id, cnt in track_id2cnt.items():
            gt_len = gt_track_len[t_id]  # length of the gt_track
            assert cnt <= gt_len
            if N == 0:
                if cnt > 0:
                    recalled_track_ids.append(t_id)
            else:
                if (cnt / gt_len) * 100 >= N:
                    recalled_track_ids.append(t_id)

        ratio = len(set(recalled_track_ids)) / len(set(all_track_ids))
        iou_curve.append(ratio)

    return iou_curve


def recall_eval(props_dir: str, gt_path: str, outdir: str, iou_threshold: float, scoring='', datasplit='val'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    """ all_proposals (Dict):
    { frame_name: List[[x1,y1,x2,y2]] }
    """
    all_proposals = load_proposals(props_dir, datasplit)

    """ gt (Dict):
    {frame_name: {"bboxes: List[[x1,y1,x2,y2]], "track_ids": List[int]}}
    """
    # --------- Known ----------
    print("evaluating coco 78 classes")
    exclude_classes = tuple(unknown_tao_ids.union(neighbor_classes))
    gt, total_boxes = load_gt(gt_path, exclude_classes=exclude_classes)
    iou_curve_known = evaluate_proposals(gt, all_proposals, iou_threshold)
    print("total_boxes:", total_boxes)
    print("iou_curve")
    print(iou_curve_known)
    try:
        np.savetxt(os.path.join(outdir, 'known_{}.csv'.format(scoring)), np.array(iou_curve_known), delimiter=',', fmt='%1.4f')
    except:
        print("unable to save csv")

    # -------- Neighbor ---------
    print("evaluating neighbor classes")
    exclude_classes = tuple(known_tao_ids.union(unknown_tao_ids))
    gt, total_boxes = load_gt(gt_path, exclude_classes=exclude_classes)
    iou_curve_neighbor = evaluate_proposals(gt, all_proposals, iou_threshold)
    print("total_boxes:", total_boxes)
    print("iou_curve")
    print(iou_curve_neighbor)
    try:
        np.savetxt(os.path.join(outdir, 'neighbor_{}.csv'.format(scoring)), np.array(iou_curve_neighbor), delimiter=',', fmt='%1.4f')
    except:
        print("unable to save csv")

    # -------- Unknown -----------
    print("evaluating unknown")
    exclude_classes = tuple(known_tao_ids.union(neighbor_classes))
    gt, total_boxes = load_gt(gt_path, exclude_classes=exclude_classes)
    iou_curve_unknown = evaluate_proposals(gt, all_proposals, iou_threshold)
    print("total_boxes:", total_boxes)
    print("iou_curve")
    print(iou_curve_unknown)
    try:
        np.savetxt(os.path.join(outdir, 'unknown_{}.csv'.format(scoring)), np.array(iou_curve_unknown), delimiter=',', fmt='%1.4f')
    except:
        print("unable to save csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script for Proposal Similarity')
    parser.add_argument('--outdir', type=str, help='Plots output dir.')
    parser.add_argument('--props_dir', type=str, help='Base directory of the proposals to be evaluated.')
    parser.add_argument('--labels', type=str, default='', help='Specify dir containing the labels')
    parser.add_argument('--score_func', type=str, help='Sorting criterion to use. Choose from' +
                                                       '[score, bg_score, 1-bg_score, obj, bg+obj, bg*obj]')
    parser.add_argument('--iou_threshold', default=0.5, type=float,
                        help='When proposal and gt-object have IoU above this IoU threshold, the gt is recalled')
    parser.add_argument('--postNMS', action='store_true', help='processing postNMS proposals.')
    parser.add_argument('--datasplit', default='val', type=str, help='evalution dataset from [train, val, test]')
    # parser.add_argument('--nonOverlap', action='store_true',
    #                     help='Filter out the overlapping bboxes in proposals using score. Proposals with higher score stay')
    # parser.add_argument('--nonOverlap_small', action='store_true',
    #                     help='Filter out the overlapping bboxes in proposals using area. Proposals with smaller area stay')
    args = parser.parse_args()

    scoring_names = ["bg_obj_prod", "bg_obj_sum", "bg_score",
                     "objectness", "one_minus_bg_score", "score"]

    if args.props_dir:
        if args.postNMS:
            all_props_dirs = [args.props_dir + '_' + sd for sd in scoring_names]
            for idx, props_dir in enumerate(all_props_dirs):
                print("Processing", props_dir)
                recall_eval(props_dir, args.labels, args.outdir, args.iou_threshold,
                            scoring=scoring_names[idx], datasplit=args.datasplit)
        else:
            props_dir = args.props_dir
            # different scoring here do not make a difference.
            recall_eval(props_dir, args.labels, args.outdir, args.iou_threshold)

    plot_from_csv(csv_folder=args.outdir, outdir=args.outdir, postNMS=args.postNMS)

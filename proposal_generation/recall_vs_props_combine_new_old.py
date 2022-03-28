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

all_ids = set([i for i in range(1, 1231)])

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
# plot functions
# =======================================================
def plot_from_csv(csv_folder1: str, csv_folder2, outdir: str, curr_scoring, postNMS: bool, n_max_props: int):
    # hard coded
    nboxes = {
        "known": {"all": 86405, "YFCC100M": 11088, "LaSOT": 19340, "HACS": 23522, "Charades": 8743,
                  "BDD": 7606, "ArgoVerse": 3621, "AVA": 12485},
        "neighbor": {"all": 5232, "YFCC100M": 671, "LaSOT": 1329, "HACS": 904, "Charades": 1177,
                     "BDD": 687, "ArgoVerse": 123, "AVA": 341},
        "unknown": {"all": 20522, "YFCC100M": 1690, "LaSOT": 6066, "HACS": 5462, "Charades": 4928,
                    "BDD": 28, "ArgoVerse": 116, "AVA": 2232}
    }

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

    # if postNMS:
    scoring_list = ["score", "bgScore", "1-bgScore", "objectness", "bg+objectness", "bg*objectness"]
    export_dict = {"score": {}, "bgScore": {}, "1-bgScore": {},
                   "objectness": {}, "bg+objectness": {}, "bg*objectness": {}}


    # else:
    #     scoring_list = [""]
    #     export_dict = {"": {}}

    if curr_scoring:
        scoring_list = [curr_scoring]
        export_dict = {curr_scoring: {}}

    print("Read in csv files")
    for split in ["known", "neighbor", "unknown"]:
        for scoring in tqdm.tqdm(scoring_list):
            # if postNMS:
            # csv_path = os.path.join(csv_folder, split + '_' + scoring2suffix(scoring) + ".csv")
            csv_path1 = os.path.join(csv_folder1, split + '_' + scoring2suffix(scoring) + ".csv")
            csv_path2 = os.path.join(csv_folder2, split + '_' + scoring2suffix(scoring) + ".csv")
            # else:
            #     csv_path = os.path.join(csv_folder, split + ".csv")
            y = list()
            with open(csv_path1, newline='') as csvfile1:
                reader1 = csv.reader(csvfile1, delimiter=' ')
                for row in reader1:
                    y.append(float(row[0]))
            export_dict[scoring]["data1"] = y

            y = list()
            with open(csv_path2, newline='') as csvfile2:
                reader2 = csv.reader(csvfile2, delimiter=' ')
                for row in reader2:
                    y.append(float(row[0]))
            export_dict[scoring]["data2"] = y

        x_vals = range(n_max_props + 1)
        total_boxes = nboxes[split]['all']
        plot_title = split + " classes (" + str(total_boxes) + " bounding boxes)"
        outdir = os.path.join(csv_folder1, "combined_final")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        make_plot(outdir, split, export_dict, plot_title, x_vals, linewidth=4, n_max_props=n_max_props)


def title_to_filename(plot_title):
    filtered_title = re.sub("[\(\[].*?[\)\]]", "", plot_title)  # Remove the content within the brackets
    filtered_title = filtered_title.replace("_", "").replace(" ", "").replace(",", "_")
    return filtered_title


def make_plot(output_dir, split, export_dict, plot_title, x_vals, linewidth=5, n_max_props=1000):
    plt.figure()

    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    itm = export_dict.items()
    itm = sorted(itm, reverse=True)
    for idx, item in enumerate(itm):
        # Compute Area Under Curve
        x = x_vals[0:n_max_props]
        y1 = item[1]['data1'][0:n_max_props]
        y2 = item[1]['data2'][0:n_max_props]
        auc1= round(metrics.auc(x, y1), 2)
        auc2= round(metrics.auc(x, y2), 2)

        auc1 = auc1 / n_max_props
        auc1 = round(auc1, 2)
        auc2 = auc2 / n_max_props
        auc2 = round(auc2, 2)

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
        curve_label = curr_scoring + ' (' + str(auc1) + ' vs ' + str(auc2) + ')'
        # curve_label1 = curr_scoring + '(new)' + ' (' + str(auc1) + ')'
        # curve_label2 = curr_scoring + '(old)' + ' (' + str(auc2) + ')'
        plt.plot(x_vals[0:n_max_props], item[1]['data1'][0:n_max_props], color=colors[idx], label=curve_label, linewidth=linewidth)
        plt.plot(x_vals[0:n_max_props], item[1]['data2'][0:n_max_props], color=colors[idx], linestyle='dashed', linewidth=linewidth)

    ax = plt.gca()
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    if n_max_props == 1000:
        ax.set_xticks(np.asarray([10, 200, 500, 700, 1000]))
    else:
        ax.set_xticks(np.asarray([1, 20, 50, 70, 100, 150, 200]))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(f"$\#$ proposals ({split})", fontsize=10)
    # plt.ylabel("Recall", fontsize=20)
    ax.set_ylim([0.0, 1.0])
    plt.legend(prop={"size": 16}, handlelength=1.0, labelspacing=0.2, borderaxespad=0.2, loc='lower right')
    plt.grid()
    # plt.title(plot_title)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script for Proposal Similarity')
    parser.add_argument('--csv_dir1', type=str, help='Plots output dir.')
    parser.add_argument('--csv_dir2', type=str, help='Plots output dir.')
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

    # If previous evaluation are completed and stored as CSV files, leave args.props_dir empty and
    # use the function below to directly ouput plots.
    plot_from_csv(csv_folder1=args.csv_dir1, csv_folder2=args.csv_dir2, outdir=args.csv_dir1, curr_scoring=None, postNMS=args.postNMS, n_max_props=args.n_max_props)


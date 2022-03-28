import argparse
import glob
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os
import pycocotools.mask as rletools
import shutil

from PIL import Image

from utils import colormap
from utils import frames2video

"""
Visualization script for tracking result storeed in MOT format.

<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>,
<img_h>, <img_w>, <rle> 
"""
"""
<frame_id> <track_id> <class_id> <img_h> <img_w> <rle>
"""


class SegmentedObject:
    def __init__(self, bbox, mask, score, class_id, track_id):
        self.bbox = bbox
        self.mask = mask
        self.score = score
        self.class_id = class_id
        self.track_id = track_id


def box_IoU_xywh(boxA, boxB):
    """input box: [x,y,w,h]"""
    # convert [x,y,w,h] to [x1,y1,x2,y2]
    xA, yA, wA, hA = boxA
    boxA = [xA, yA, xA + wA, yA + hA]
    xB, yB, wB, hB = boxB
    boxB = [xB, yB, xB + wB, yB + hB]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def load_txt_tao(path):
    """
    0<frame>, 1<id>, 2<bb_left>, 3<bb_top>, 4<bb_width>, 5<bb_height>, 6<conf>, 7<x>, 8<y>, 9<z>
    10<img_h>, 11<img_w>, 12<rle>
    """

    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            # class_id = int(fields[2])
            class_id = 1
            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            fields[12] = fields[12].strip()
            score = float(fields[6])
            mask = {'size': [int(fields[10]), int(fields[11])], 'counts': fields[12].encode(encoding='UTF-8')}
            bbox = [float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])]  # [x, y, w, h]

            objects_per_frame[frame].append(SegmentedObject(
                bbox=bbox,
                mask=mask,
                score=score,
                class_id=class_id,
                track_id=int(fields[1])
            ))

    return objects_per_frame


def load_sequences(seq_paths):
    objects_per_frame_per_sequence = {}
    print("Loading Sequences")
    for seq_path_txt in tqdm.tqdm(seq_paths):
        seq = seq_path_txt.split("/")[-1].replace(".txt", "")
        if os.path.exists(seq_path_txt):
            objects_per_frame_per_sequence[seq] = load_txt_tao(seq_path_txt)
            # Sort proposals in each frame by score
            for frame_id, props in objects_per_frame_per_sequence[seq].items():
                props.sort(key=lambda p: p.score, reverse=True)
        else:
            assert False, "Can't find data in directory " + seq_path_txt

    return objects_per_frame_per_sequence


def process_all_sequence(track_results_fpaths, gt_frame2anns, annot_frames, img_folder, output_folder, topN_proposals):
    os.makedirs(output_folder, exist_ok=True)
    for seq_fpath in tqdm.tqdm(track_results_fpaths):
        video_name = seq_fpath.split('/')[-1].replace(".txt", "")
        tracks = load_sequences([seq_fpath])

        img_dir = os.path.join(img_folder, video_name)
        # frame_paths = sorted(glob.glob(img_dir + '/*' + '.jpg'))
        # if not frame_paths:
        #     frame_paths = sorted(glob.glob(img_dir + '/*' + '.png'))
        frame_paths = annot_frames[video_name]

        visualized_one_sequence(tracks, video_name, frame_paths, output_folder, draw_boxes=True, create_video=True)


def visualized_one_sequence(tracks, video_name, frame_paths, outdir, draw_boxes=False, create_video=True):
    colors = colormap()
    dpi = 100.0

    frame_id2dets = tracks[video_name]

    for i, fp in enumerate(frame_paths[:50]):
        frame_name = fp.split('/')[-1].replace(".png", '').replace(".jpg", '')
        img = np.array(Image.open(fp), dtype="float32") / 255
        img_sizes = img.shape

        dets = frame_id2dets.get(i+1, [])

        fig = plt.figure()
        fig.set_size_inches(img_sizes[1] / dpi, img_sizes[0] / dpi, forward=True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax = fig.subplots()
        ax.set_axis_off()

        for obj in dets:
            color = colors[obj.track_id % len(colors)]
            category_id = str(obj.class_id)

            if draw_boxes:
                # x, y, w, h = rletools.toBbox(obj.mask)
                x, y, w, h = obj.bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                         edgecolor=color, facecolor='none', alpha=1.0)
                ax.add_patch(rect)

            binary_mask = rletools.decode(obj.mask)
            apply_mask(img, binary_mask, color)

        ax.imshow(img)
        if not os.path.exists(os.path.join(outdir + "/" + video_name)):
            os.makedirs(os.path.join(outdir + "/" + video_name))
        fig.savefig(outdir + "/" + video_name + "/" + frame_name + '.png')
        plt.close(fig)


    if create_video:
        fps = 10
        frames2video(pathIn=outdir + "/" + video_name + '/',
                     pathOut=outdir + "/" + video_name + ".mp4", fps=fps)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualization script for tracking result')
    parser.add_argument("--tracks_folder", type=str,
                        help="Folder contains tracking result. Should be combined with --datasrc")
    parser.add_argument("--gt_path", type=str, help="File path to GT file.")
    parser.add_argument("--img_folder", type=str, help="Folder contains raw images. Should be combined with --datasrc")
    parser.add_argument("--datasrc", type=str,
                        choices=["ArgoVerse", "AVA", "BDD", "Charades", "LaSOT", "YFCC100M", "HACS", "image_02"])
    # parser.add_argument("--phase", default="objectness", help="objectness, score or one_minus_bg_score", type=str)
    # parser.add_argument("--only_annotated", action="store_true")
    # parser.add_argument("--tao_subset", action="store_true", help="if only process fixed tao-subsets")
    parser.add_argument("--topN_proposals", default="1000",
                        help="for each frame, only display top N proposals (according to their scores)", type=int)
    args = parser.parse_args()

    tracks_folder = os.path.join(args.tracks_folder, args.datasrc)
    img_folder = os.path.join(args.img_folder, args.datasrc)
    output_folder = os.path.join(args.tracks_folder, "viz_" + str(args.topN_proposals), args.datasrc)

    topN_proposals = args.topN_proposals
    print("For each frame, display top {} proposals".format(topN_proposals))

    # Pre-processing GT
    # gt_frame_name2anns, tao_id2name = load_and_preprocessing_gt(args.gt_path, args.datasrc)

    annot_frames = dict()  # annotated frames for each sequence

    # Get the annotated frames in the current sequence.
    txt_fname = "../datasets/val_annotated_{}.txt".format(args.datasrc)
    with open(txt_fname) as f:
        content = f.readlines()
    content = ['/'.join(c.split('/')[2:]) for c in content]
    annot_seq_paths = [os.path.join(img_folder, x.strip()) for x in content]

    for s in annot_seq_paths:
        seq_name = s.split('/')[-2]
        if seq_name not in annot_frames.keys():
            annot_frames[seq_name] = []
        annot_frames[seq_name].append(s)

    # videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(img_folder, '*')))]
    # for video_name in videos:
    #     video_path = os.path.join(img_folder, video_name)
    #     frames = sorted(glob.glob(video_path + '/*' + '.jpg'))
    #     annot_frames[video_name] = sorted(frames)

    # max_frames = dict()
    # for seq_name, frames in annot_frames.items():
    #     max_frames[seq_name] = len(frames)

    track_results_fpaths = sorted(glob.glob(tracks_folder + '/*' + '.txt'))
    process_all_sequence(track_results_fpaths, None, annot_frames, img_folder, output_folder, topN_proposals)

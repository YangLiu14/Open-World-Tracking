import argparse
import cv2
import glob
import json
import numpy as np
import os
import pycocotools.mask as rletools
import tqdm
import warnings

from pycocotools.mask import toBbox


class SegmentedObject:
    def __init__(self, mask, class_id, track_id):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


# ===============================================
# TAO-MOTS20 dataset utils
# ===============================================
def load_seqmap(seqmap_filename):
    print("Loading seqmap...")
    seqmap = []
    max_frames = {}
    with open(seqmap_filename, "r") as fh:
        for i, l in enumerate(fh):
            fields = l.split(" ")
            seq = "%04d" % int(fields[0])
            seqmap.append(seq)
            max_frames[seq] = int(fields[3])
    return seqmap, max_frames


def load_sequences(path, seqmap):
    objects_per_frame_per_sequence = {}
    for seq in seqmap:
        print("Loading sequence", seq)
        seq_path_folder = os.path.join(path, seq)
        seq_path_txt = os.path.join(path, seq + ".txt")
        if os.path.exists(seq_path_txt):
            objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
        # elif os.path.isdir(seq_path_folder):
        #     objects_per_frame_per_sequence[seq] = load_images_for_folder(seq_path_folder)
        else:
            assert False, "Can't find data in directory " + path

    return objects_per_frame_per_sequence


def load_txt(path):
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not(class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
                assert False, "Objects with overlapping masks in frame " + fields[0]
            else:
                combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)
            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame


# ===============================================
# TAO-MOT dataset utils
# ===============================================
def load_tao_gt(gt_path: str, image_dir: str):
    """
    Get the corresponding {image_id, video_id, frame_idx} to each image_name.
    Get all the frames in each video
    Args:
        gt_path: json ground truth file path.
        image_dir: directory of images.
    Returns:
        Dict, Dict: image_name2ids, frames_in_video
    """
    with open(gt_path, 'r') as f:
        gt_dict = json.load(f)

    gt_images = gt_dict["images"]

    image_name2ids = dict()
    frames_in_video = dict()

    for img in gt_images:
        img_name = '/'.join(img["file_name"].split("/")[1:])[:-4]
        if img_name not in image_name2ids.keys():
            image_name2ids[img_name] = ''
        image_name2ids[img_name] = {'image_id': img["id"], "video_id": img["video_id"], "frame_idx": img["frame_index"]}

        video_name = img["video"].split("/", 1)[1]
        frame_name = img_name.split('/')[-1]
        if video_name not in frames_in_video.keys():
            frames_in_video[video_name] = {"frame_idxs": [], "frame_names": []}
        frames_in_video[video_name]["frame_idxs"].append(img["frame_index"])
        frames_in_video[video_name]["frame_names"].append(frame_name)

    return image_name2ids, frames_in_video


def convert_mot2tao_results(mot_results_dir:str, gt_path: str, image_dir: str,
                            outdir: str, outname: str, split: str, only_annotated=True):
    """
    Convert tracking results in mot-format to Tao-format.
    - MOT-format:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    - TAO-format:
        [{
            "image_id" : int,
            "category_id" : int,
            "bbox" : [x,y,width,height],
            "score" : float,
            "track_id": int,
            "video_id": int
        }]
    - mot_results_dir folder structure:
        mot_results_dir/
            ArgoVerse/
                video1.txt
                video2.txt
                ...
            BDD/
            Charades/
            LaSOT/
            YFCC100M/
    """
    track_results = list()

    image_name2ids, frames_in_video = load_tao_gt(gt_path, image_dir)

    data_srcs = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(mot_results_dir, '*')))]

    for data_src in data_srcs:
        print("Processing", data_src)
        fpath = os.path.join(mot_results_dir, data_src)
        files = sorted(glob.glob(fpath + '/*' + '.txt'))

        if only_annotated:
            # Load sequence of annotated frames
            txt_fname = "./datasets/{}_annotated_{}.txt".format(split, data_src)
            with open(txt_fname) as f:
                content = f.readlines()
            annot_frames = dict()
            for c in content:
                c = c.strip()
                video_name = data_src + '/' + c.split('/')[-2]
                frame_name = c.split('/')[-1].replace(".jpg", "")
                if video_name not in annot_frames.keys():
                    annot_frames[video_name] = list()
                annot_frames[video_name].append(frame_name)
                annot_frames[video_name].sort()

        for txt_file in tqdm.tqdm(files):
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>,
            # <img_h>, <img_w>, <rle>
            with open(txt_file, 'r') as txt_f:
                content = txt_f.readlines()
            content = [c.strip() for c in content]
            mot_results = np.array([line.split(',') for line in content])
            frame_ids = mot_results[:, 0].astype(int)
            max_frames = np.max(frame_ids)

            # Process proposals in each frame and store them in TAO-format
            video_name = data_src + '/' + txt_file.split('/')[-1].replace(".txt", "")
            if video_name not in frames_in_video.keys():
                print(video_name, "not in the keys of frames_in_video")
                continue

            frame_id = 1
            while frame_id <= max_frames:
                # Get image_id, video_id from GT
                # Get frame name through frame_id
                if only_annotated:  # when mot-tracking results only contains annotated frames
                    frame_name = annot_frames[video_name][frame_id - 1]
                else:
                    if (frame_id - 1) not in frames_in_video[video_name]["frame_idxs"]:
                        frame_id += 1
                        continue
                    else:
                        idx = frames_in_video[video_name]["frame_idxs"].index(frame_id - 1)
                        frame_name = frames_in_video[video_name]["frame_names"][idx]

                image_id = image_name2ids[video_name + '/' + frame_name]["image_id"]
                video_id = image_name2ids[video_name + '/' + frame_name]["video_id"]

                # proposals_per_frame = mot_results[mot_results[:, 0] == frame_id]
                proposals_per_frame = [prop[:10] for prop in mot_results if int(prop[0]) == frame_id]  # also cut out mask
                proposals_per_frame = np.array(proposals_per_frame, dtype=np.float32)

                if len(proposals_per_frame) > 0:
                    for prop in proposals_per_frame:
                        curr = {"image_id": int(image_id),
                                "category_id": 1,
                                "bbox": prop[2:6].tolist(),  # [x, y, w, h]
                                "score": float(prop[6]),
                                "track_id": int(prop[1]),
                                "video_id": int(video_id)
                                }
                        track_results.append(curr)
                frame_id += 1
    # Store track_result in TAO-format to json file
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, outname + '.json'), 'w') as f2:
        json.dump(track_results, f2)


if __name__ == "__main__":
    # =========================================================
    # Convert Tracking-results in MOT-format to COCO-format
    # =========================================================
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--results_dir', type=str, help='Directory where tracking results (in mot format) are')
    parser.add_argument('--gt_path', type=str, help='Directory where the GT annotation file is (json)')
    parser.add_argument('--image_dir', type=str, help='Image folder', default="no need")
    parser.add_argument('--outname', type=str, help='Output file name.')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val')

    args = parser.parse_args()

    results_dir = args.results_dir
    gt_path = args.gt_path
    image_dir = args.image_dir
    outdir = results_dir + "_converted"
    outname = args.outname

    convert_mot2tao_results(results_dir, gt_path, image_dir, outdir, outname, args.split, only_annotated=True)

import argparse
import glob
import json
import numpy as np
import os
import tqdm

from pycocotools.mask import encode, decode, area, toBbox


def remove_mask_overlap(proposals, scoring=None):
    """
    Args:
        proposals: List[Dict], each proposal contains:
        {
            category_id: int,
            bbox: [x1, y1, x2, y2],
            score: float (could be named differently, e.g. bg_score, objectness, etc)
            instance_mask: COCO_RLE format,
        }

        scoring: str, current scoring strategies the proposals are using, e.g. "score", "objectness", "bg_score", etc.

    Returns:
        selected_props, List[Dict]
    """
    if scoring:
        proposals.sort(key=lambda prop: prop["score"])   # [lower_score-proposal, ..., higher_score-proposal]
    else:
        proposals.sort(key=lambda prop: area(prop['instance_mask']), reverse=True)

    masks = [decode(prop['instance_mask']) for prop in proposals]
    idx = [i for i in range(len(proposals))]
    labels = np.arange(1, len(proposals) + 1)
    png = np.zeros_like(masks[0])

    # When there are overlapping pixels, assign those pixels to the masks the comes latter,
    # here it means those masks with higher-scores, or with smaller-areas.
    for i in range(len(proposals)):
        png[masks[i].astype("bool")] = labels[i]

    refined_masks = [(png == id_).astype(np.uint8) for id_ in labels]
    refined_segmentations = [encode(np.asfortranarray(refined_mask)) for refined_mask in refined_masks]
    selected_props = []
    for prop, refined_segmentation, mask in zip(proposals, refined_segmentations, refined_masks):
        refined_segmentation['counts'] = refined_segmentation['counts'].decode("utf-8")
        if area(refined_segmentation) == 0:
            continue
        prop['instance_mask'] = refined_segmentation
        box = toBbox(refined_segmentation).tolist()  # in the form of [xc, yc, w, h]
        # convert [xc, yc, w, h] to [x1, y1, x2, y2]
        bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        prop['bbox'] = bbox

        selected_props.append(prop)

    return selected_props


def process_one_frame(fpath:str, outdir:str, scoring: str, criterion, file_type='.json'):
    if file_type == ".json":
        with open(fpath, 'r') as f:
            proposals = json.load(f)
    elif file_type == ".npz":
        proposals = np.load(fpath, allow_pickle=True)['arr_0'].tolist()
    else:
        raise Exception("unrecognized file type.")

    if criterion == "score":
        processed = remove_mask_overlap(proposals, scoring)
    elif criterion == "area":
        processed = remove_mask_overlap(proposals)
    else:
        raise Exception("%s is not a valid criterion for Non-overlap operation" % criterion)

    # Store processed proposals
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    frame_name = fpath.split('/')[-1]
    if file_type == ".json":
        with open(os.path.join(outdir, frame_name), 'w') as f1:
            json.dump(processed, f1)
    elif file_type == ".npz":
        np.savez_compressed(outdir, processed)
    else:
        raise Exception("Invalid file type.")


def main(input_dir: str, outdir: str, scoring: str, criterion: str, file_type='.json'):
    videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(input_dir, '*')))]
    for idx, video in enumerate(videos):
        print("{}/{} processing {}".format(idx+1, len(videos), video))
        fpath = os.path.join(input_dir, video)
        frames = sorted(glob.glob(fpath + '/*' + file_type))
        for frame_fpath in tqdm.tqdm(frames):
            process_one_frame(frame_fpath, outdir + '/' + video, scoring, criterion, file_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", help="Input directory")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument("--criterion", help="higher score on top or smaller area on top")
    parser.add_argument("--scoring", required=True, help="scoring criterion used tp produce the NMS result")
    parser.add_argument('--datasrcs', nargs='+', type=str, help='IoU threshold used in NMS')
    parser.add_argument('--file_type', default=".json", type=str, help='.npz or .json')
    args = parser.parse_args()

    for datasrc in args.datasrcs:
        print("Current data source:", datasrc)
        input_dir = os.path.join(args.inputdir, datasrc)
        if args.criterion == "score":
            outdir = os.path.join(args.outdir, "high_score_on_top", '_' + args.scoring, datasrc)
        elif args.criterion == "area":
            outdir = os.path.join(args.outdir, "small_area_on_top", '_' + args.scoring, datasrc)
        main(input_dir, outdir, args.scoring, args.criterion, file_type=args.file_type)

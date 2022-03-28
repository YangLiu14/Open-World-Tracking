"""
get_annot_frames.py
Iterate through annotations and get all the annotated frames.
"""
import argparse
import json
import os
import os.path as osp


def get_annot_frames(fpath, outdir):
    with open(fpath, 'r') as f:
        gt_dict = json.load(f)

    annot_frames_per_datasrc = {
        'ArgoVerse': set(),
        'AVA': set(),
        'BDD': set(),
        'Charades': set(),
        'HACS': set(),
        'LaSOT': set(),
        'YFCC100M': set()
    }

    for gt in gt_dict['images']:
        img_name = gt['file_name']
        curr_datasrc = img_name.split(os.sep)[1]
        annot_frames_per_datasrc[curr_datasrc].add(img_name)

    for datasrc, frame_names in annot_frames_per_datasrc.items():
        frame_names = sorted(list(frame_names))

        split =  frame_names[0].split(os.sep)[0]
        outpath = osp.join(outdir, '_'.join([split, 'annotated', datasrc]) + '.txt')
        with open(outpath, 'w') as fout:
            for line in frame_names:
                fout.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Annotated frames for each data source')
    parser.add_argument('--root', type=str, help='Root dir that contains the annotations files')
    parser.add_argument('--split', type=str,
                        choices=['train', 'train_with_freeform', 'val', 'val_with_freeform', 'test'])
    parser.add_argument('--outdir', type=str, help='The annotated frames would be output as txt file.')
    args = parser.parse_args()

    fname = ''
    if args.split == 'train':
        fname = "train.json"
    elif args.split == 'train_with_freeform':
        fname = "train_with_freeform.json"
    elif args.split == 'val':
        fname = "validation.json"
    elif args.split == 'val_with_freeform':
        fname = "validation_with_freeform.json"
    elif args.split == 'test':
        fname = "test_without_annotations.json"

    if fname:
        fpath = osp.join(args.root, fname)
        os.makedirs(args.outdir, exist_ok=True)
        get_annot_frames(fpath, args.outdir)

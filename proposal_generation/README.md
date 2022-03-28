# Proposal Generation

We adapt **detectron2** to generate per-frame proposals. (Code will be published soon)

Each proposal we generate contains the following information:

- `bbox`: [x1, y1, x2, y2]
- `instance_mask`: RLE format
- `score`: the standard score of the most confident class among the 80 COCO classes. 
- `bg_score`: defined as $1 - \sum_c score(c)$​​, where sum is over all 80 COCO classes. This measures how much the object is **not** not one of the COCO classes.
- `objectness`: the proposal score directly from Region Proposal Network. 
- `category_id`
- `embeddings`: 1x1024 vector, extracted from the FC-layers in the very last part of Mask RCNN. This serves as **apperance embeddings**. Originally, each value in the vector is in the range of [0, 1]. To save storage space, we multiply each value by `1e4`, and convert to `uint16`. When using the embeddings, it is suggested to divide the vector by `1e4` to obtain values between [0, 1].

 ## Evaluation
 ### Recall Analysis
`eval_recall_vs_props.py`
 
All the proposals (per frame) are sorted by different scorings (objectness, background-score, etc.). 
Given `n` proposals (per frame), `n` is in the range of [0, 1000] in our experiment, we evaluation how much percentage of the
Ground Truth objects that the `n` proposals can recall. And plot the curve as shown in the Figure 5 (left) in our paper.

`eval_recall_vs_tracks.py`
Evalute the percentage of tracks recalled over different minimum relative track lengths required for recall. 
For more details please see Figure 5 (right) in our paper.

`stats_bbox_size.py`
Analyse the influence of bbox size on Recall.

#### Examples of usage
```shell script
python proposal_generation/eval_recall_vs_props.py \
  --outdir <output directory> \
  --props_dir <input directory> \
  --labels <path to GT.json file> \
  --datasplit val \
  --postNMS --n_max_props 1000
```

```shell script
python proposal_generation/eval_recall_vs_tracks.py \
  --outdir <output directory> \
  --props_dir <input directory> \
  --labels <path to GT.json file> \
  --datasplit val
```

```shell script
python proposal_generation/stats_bbox_size.py \
 --props_dir <directory containing proposals> \
 --gt_path <path to GT.json file> \
 --outdir <output directory> \
 --bound_large 0.3 --bound_small 0.03 \
 --split unknown
```


## Things worth mentioning
1. The bbox annotations in TAO are in the format of `[x, y, w, h]`, `w`: width, `h`: height.
while the bboxes directly output from our detector are in the format of `[x1, y1, x2, y2]`. 

   
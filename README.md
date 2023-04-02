# Open-World-Tracking
Official code for "Opening up Open-World Tracking" (CVPR 2022 Oral) 

[Project Page](https://openworldtracking.github.io/)

![teaser](assets/teaser.png)


The repository contains the code for the experiments we conducted
in the paper. We also provide the baseline trackers.

<pre><b>Opening up Open-World Tracking</b>
Yang Liu*, Idil Esen Zulfikar*, Jonathon Luiten*, Achal Dave*, Deva Ramanan, Bastian Leibe, Aljoša Ošep, Laura Leal-Taixé
<t><t>*Equal contribution
CVPR 2022 Oral</pre>

## 1. Dependencies
- Python >= 3.6
- PyTorch==1.5
- detectron2 installation, see [this](https://github.com/YangLiu14/detectron2-OWT)

You can also check `requirements.txt` for more details.


## 2. Data and Models

### Data Download
We use TAO as our base dataset, so please follow the instructions [here](https://motchallenge.net/tao_download.php) 
to download the raw data and the labels.

### Known and Unknown Split
As stated in the paper, we use COCO-classes as known, you can find the mapping between
`coco_id` to `tao_id` in `datasets/coco_id2tao_id.json`.
Keeping the default folder structure is fine.



To get the known/unknown ids in TAO:
```python
import json

json_path = "./datasets/coco_id2tao_id.json"
with open(json_path, 'r') as f:
    coco2tao_map = json.load(f)
knowns = {v for _, v in coco2tao_map.items()}

json_path = "./datasets/distractor_classes.json"
with open(json_path, 'r') as f:
    coco2distractor_map = json.load(f)

distractors = []
for _, v in coco2distractor_map.items():
    distractors += v
distractors_ids = set(distractors)

all_ids = set([i for i in range(1, 2000)])  # 2000 is larger than the max category id in TAO-OW.
unknowns = all_ids.difference(knowns.union(distractors))
```

### Model
We adapt **detectron2** to generate per-frame proposals.
Please visit https://github.com/YangLiu14/detectron2-OWT

The detailed instructions/model config/model weights are in 
[README](https://github.com/YangLiu14/detectron2-OWT/blob/master/README.md) 
in the repo mentioned above.

## 3. Running the Tracker
The detailed instruction is in `trackers/README.md`.
Give the correct information in `config.yaml` and run:
```
python nonoverlap_track.py --tracker online-ghost --config ./configs/example_tracker.yaml
```
The tracking results will be written as `.txt` files.

For evaluation, you need to aggregate the results in all the txt files in 
one single `.json` file.

You can use `tools/track_results_conversion.py` to do that.
```shell
python tools/track_result_conversion.py \
  --results_dir /results/my_tracker/ \
  --gt_path /data/TAO/annotations/validations.json \
  --outname tracking \
  --split val
```

After running this script, you will get `/results/my_tracker/_COCO/tracking.json`


## 4. Evaluation
For instructions for evaluation, please visit [link](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/OpenWorldTracking-Official/Readme.md)

1. download/clone the `TrackEval` 
```
git clone https://github.com/JonathonLuiten/TrackEval.git
```
2. Copy the GT annotation from `/data/TAO/annotations/validation.json` to a new folder (say `/data/TAO/trackeval/validations.json`).
Make sure this folder only contains **one** annotation file.

3. Suppose your tracking results (COCO format, see section 3) are stored in `/results/my_tracker/_COCO/tracking.json`
```shell
cd scripts/

python run_tao_ow.py \
  --GT_FOLDER /data/TAO/trackeval/ \
  --TRACKERS_FOLDER results \
  --TRACKERS_TO_EVAL my_trackers \
  --SPLIT_TO_EVAL val \
  --TRACKER_SUB_FOLDER _COCO \
  --TRACKER_DISPLAY_NAMES MY_TRACKER \
  --MAX_DETECTIONS 1000 \
  --SUBSET unknown
```


## Citation
```
@inproceedings{liu2022opening,
  title={Opening up Open-World Tracking},
  author={Liu, Yang and Zulfikar, Idil Esen and Luiten, Jonathon and Dave, Achal and Ramanan, Deva and Leibe, Bastian and O{\v{s}}ep, Aljo{\v{s}}a and Leal-Taix{\'e}, Laura},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
# List of Tools

1. Non-maxima Supression: `nms.py`

## NMS(Non-maxima Supression)

`nms.py`

Assuming you generated proposals with NMS disabled like we did in the paper, then for each frame you have 1000 proposals. We provided a seperate script that allows you to perform NMS based on different scoring strategies. We implemented the following scoring stragies:

- `score`:  this is just the original NMS operation.
- `bg_score`: keep proposals with higher background scores.
- `one_minus_bg_score`: keep proposals with higher `1 - bg_score`.
- `objectness`: keep proposals with higher objectness score
- `bg_obj_sum`: treat the sum of `bg_score` and `objectness` as a new score. 
- `bg_obj_prod`: treat the product of `bg_score` and `objectness` as a new score. 

It's also possible for you to implemennt more combinations of difference scoring strategies.

#### Example of usage

```cd proposal_generation```

```bash
python nms.py --inputdir <input directory containing proposals> \
              --outdir <output directory> \
              --datasrcs ArgoVerse Charades LaSOT \
              --scorings objectness score one_minus_bg_score \
              --iou_thres 0.5
```

## Non-overlap
`non_overlap.py`

Remove overlaps among proposals (per frame).

There are two strategies for overlap removal. 
The most common one assigns overlapping pixels to masks that have higher scores.
The second strategy assigns overlapping pixels to masks that have smaller areas. 





# Tracking

## Non-overlap then track
In this part we perform non-overlapping operations on the proposals per frame, then perform tracking.
We provide 3 trackings methods: **online**, **online-ghost**, **offline**.
- **online**: only perform proposal-matching between continuous frames, end a track immediately when there is no matching
proposal in the next frame.
- **online-ghost**: is built upon **online tracker**, it keeps a track "alive" for `ka_frames`, if the track matches with
any proposals within `ka_frames`, the track is extended. The parameter `ka_frames` can be tuned in the `configs/example_tracker.yaml`.
- **offline**: offline tracker is built upon **online tracker**, which online tracker provide `tracklets` that only
connects prooposals between continuous frames, **offline track** perform global optimization to connect two or more
`tracklets` together to form a longer track.


### Run
Give the information of your pre-processing proposals in `configs/example_tracker.yaml` and run:
```shell
python nonoverlap_track.py --tracker online-ghost --config ./configs/example_tracker.yaml
```
The configs set in `configs/example_tracker.yaml` is able to reproduce the OWTB (Open World Tracking Baseline) 
tracking results as shown in the paper. 


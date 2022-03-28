import glob
import json
import numpy as np
import os

import tqdm
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from association_similarity.similarity_funcs import *
from tools.non_overlap import remove_mask_overlap
from base_tracker import BaseTracker


class OnlineTracker(BaseTracker):
    def __init__(self, config):
        BaseTracker.__init__(self, config)

    @property
    def outdir(self):
        return os.path.join(self.outroot, "online", self.scoring)


class OnlineGhostTracker(BaseTracker):
    def __init__(self, config):
        BaseTracker.__init__(self, config)
        self.ka_frames = config["tracker"]["ka_frames"]

    @property
    def outdir(self):
        return os.path.join(self.outroot, "online-ghost", self.scoring)

    def keep_alive(self, t, curr_proposals):
        if len(self.tracklets) != 0:
            if t < self.ka_frames:
                n_frames_range = t
            else:
                n_frames_range = self.ka_frames
            ghost_tracklet_indices = list(
                np.where(np.array(self.tracklets)[:, t] != -1)[0])  # the active tracklets in time t are already a ghost
            for n_idx in range(0, n_frames_range):
                ghost_candidates = np.array(self.tracklets)[:, t - n_idx]  # find the tracklets that might be a ghost
                ghost_idxes = np.where(ghost_candidates == -1)[0]
                if ghost_idxes.size != 0:
                    for g_idx in ghost_idxes:
                        if g_idx not in ghost_tracklet_indices:  # if this tracklet is not already in the ghost tracklets
                            id = np.array(self.tracklets)[g_idx, t - (n_idx + 1)]
                            if id != -1:
                                ghost_prop = self.all_proposals[id]
                                curr_proposals.append(ghost_prop)
                                ghost_tracklet_indices.append(
                                    g_idx)  # add the tracklet into ghost_tracklet list to not check again

    def _run(self, datasrc, video):
        self.datasrc = datasrc
        self.video = video
        proposals_per_frames = self.load_proposals(datasrc, video)
        frames = sorted(list(proposals_per_frames.keys()))
        pairs = [(frame1, frame2) for frame1, frame2 in zip(frames[:-1], frames[1:])]

        self.all_proposals = {p['id']: p for _, props in proposals_per_frames.items() for p in props}

        self.tracklets = list()
        new_tracklet = np.negative(np.ones(len(frames), dtype=int))

        for t, (curr_frame, next_frame) in enumerate(tqdm.tqdm(pairs)):
            curr_proposals = remove_mask_overlap(proposals_per_frames[curr_frame], self.scoring)
            next_proposals = remove_mask_overlap(proposals_per_frames[next_frame], self.scoring)
            curr_all_props_id = [cp['id'] for cp in curr_proposals]

            self.keep_alive(t, curr_proposals)

            curr_props_id, next_props_id = \
                self.inter_frames_matching(curr_proposals, next_proposals, curr_frame, next_frame)

            # initialize tracklets
            if len(self.tracklets) == 0:
                if len(curr_props_id) == 0:
                    continue
                else:
                    for i, curr_id in enumerate(curr_props_id):
                        nw = new_tracklet.copy()
                        nw[t] = curr_id
                        self.tracklets.append(nw)

            if len(curr_props_id) > 0:
                for i, (curr_id, next_id) in enumerate(zip(curr_props_id, next_props_id)):
                    s = np.where(np.array(self.tracklets) == curr_id)[0]
                    if s.size != 0:  # update the active tracklet
                        self.tracklets[s[0]][t + 1] = next_id
                    else:  # new tracklet
                        nw = new_tracklet.copy()
                        nw[t] = curr_id
                        nw[t + 1] = next_id
                        self.tracklets.append(nw)

            if len(curr_all_props_id) > 0:
                active_tracklets = np.array(self.tracklets)[:, t]
                for i, curr_id in enumerate(curr_all_props_id):
                    if curr_id not in active_tracklets:
                        nw = new_tracklet.copy()
                        nw[t] = curr_id
                        self.tracklets.append(nw)

        self.tracks = self.tracklets
        return self.tracklets, proposals_per_frames, self.all_proposals
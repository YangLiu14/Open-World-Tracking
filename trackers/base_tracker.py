import glob
import json
import numpy as np
import os

import tqdm
from scipy.optimize import linear_sum_assignment

from association_similarity.similarity_funcs import *
from tools.non_overlap import remove_mask_overlap


class BaseTracker(object):
    def __init__(self, c):
        self.dataset_name = c["data"]["name"]
        self.image_dir = c["data"]["image_dir"]
        self.props_dir = c["data"]["props_dir"]
        self.opt_flow_dir = c["data"]["opt_flow_dir"]
        self.datasplit = c["data"]["datasplit"]
        self.outroot = c["data"]["outdir"]
        self.ftype = c["data"]["ftype"]
        self.datasrc, self.video = None, None

        self.scoring = c["tracker"]["scoring"]
        self.non_overlap = c["tracker"]["non_overlap"]
        if self.non_overlap == "area":
            self.scoring = None
        self.similarity = c["tracker"]["similarity"]
        self.hungarian_threshold = c["tracker"]["hungarian_threshold"]

    @property
    def outdir(self):
        return ""

    def run(self, datasrc, video):
        self._run(datasrc, video)
        self.write2txt()

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

            if self.tracklets:
                active_tracklets = np.array(self.tracklets)[:, t]

            if len(curr_props_id) > 0:
                for i, (curr_id, next_id) in enumerate(zip(curr_props_id, next_props_id)):
                    if curr_id in active_tracklets:  # update the active tracklet
                        s = np.where(np.array(self.tracklets)[:, t] == curr_id)[0][0]
                        self.tracklets[s][t + 1] = next_id
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

    def inter_frames_matching(self, curr_proposals, next_proposals, curr_frame, next_frame):
        """
        Match proposals in frame t with proposals in frame (t+1)
        Returns:
            Tuple of match ids in frame t and frame (t+1)
        """
        mat, maximize = self.similarity_association(self.similarity, curr_proposals, next_proposals,
                                                    curr_frame, next_frame)
        row_ind, col_ind = self.hungarian_matching(mat, threshold=self.hungarian_threshold, maximize=maximize)

        curr_props_id = [curr_proposals[r_idx]['id'] for r_idx in row_ind]
        next_props_id = [next_proposals[c_idx]['id'] for c_idx in col_ind]
        return curr_props_id, next_props_id

    def similarity_association(self, similarity, curr_props, next_props, curr_frame, next_frame):
        maximize = True
        mat = np.zeros((len(curr_props), len(next_props)))
        if similarity == "bbox_iou":
            pass
            # mat = np.array([
            #     similarity_iou(curr_p, next_props, curr_frame, next_frame, ) for curr_p in curr_props
            # ])
        elif similarity == "mask_iou":
            pass
        elif similarity == "optical_flow":
            pass
        elif similarity == "reid":
            pass
            # mat = np.array(
            #     [distance_reid(curr_prop, next_proposals, embedding_to_key(embedding), distance) for curr_prop in
            #      curr_proposals])
            # maximize = False
        elif similarity == "mix":
            mat = np.array(
                [similarity_mix(curr_prop, next_props, curr_frame, next_frame,
                                os.path.join(self.opt_flow_dir, self.datasrc, self.video))
                 for curr_prop in curr_props])

        return mat, maximize

    def hungarian_matching(self, mat, threshold=None, maximize=False):
        row_ind, col_ind = linear_sum_assignment(mat, maximize=maximize)

        if threshold is None:
            return row_ind, col_ind
        else:
            if not isinstance(threshold, float) and not isinstance(threshold, int):
                raise TypeError("Threshold value must be an integer or a float. Got {0}".format(type(threshold)))

            row_ind_thr = []
            col_ind_thr = []
            if maximize:
                mat = np.negative(mat)
                threshold = threshold * -1
            for r, c in zip(row_ind, col_ind):
                if mat[r][c] < threshold:
                    row_ind_thr.append(r)
                    col_ind_thr.append(c)
            return np.array(row_ind_thr), np.array(col_ind_thr)

    def load_proposals(self, datasrc, video):
        """
        Load proposals for one video sequence of the current datasrc.
        Args:
            datasrc: str
            video: str, name of the current video sequence.
            ftype: str, file type of the proposals

        Returns:
            props_per_frames: dict.
                {frame_name: List of proposals}
        """
        if self.dataset_name == "TAO":
            # TAO dataset is annotated (approximately) every 30 frames, here we only track annotated frames.
            anns_dir = os.path.join("./datasets/", self.datasplit + "_annotated_" + datasrc + '.txt')

            with open(anns_dir, 'r') as f:
                content = f.readlines()
            frame_paths = list()
            for line in content:
                line = line.strip()
                vname = line.split('/')[-2]
                if vname == video:
                    frame_name = line.split('/')[-1].replace(".jpg", "").replace(".png", "")
                    fpath = os.path.join(self.props_dir, datasrc, video, frame_name + '.' + self.ftype)
                    frame_paths.append(fpath)
            frame_paths.sort()
        else:
            vdir = os.path.join(self.props_dir, datasrc, video)
            frame_paths = sorted(glob.glob(vdir + '/*.' + self.ftype))

        props_per_frames = dict()
        start_id = 0
        for fpath in frame_paths:
            frame_name = fpath.split('/')[-1].replace('.' + self.ftype, '')
            if self.ftype == "npz":
                proposals = np.load(fpath, allow_pickle=True)['arr_0'].tolist()
            elif self.ftype == "json":
                with open(fpath, 'r') as f:
                    proposals = json.load(f)
            else:
                raise Exception("File type not supported")
            props_per_frames[frame_name] = self._preprocess_props(proposals, start_id)
            start_id += len(proposals)

        return props_per_frames

    def write2txt(self):
        """
        Write to txt in the following format
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>,
        <img_h>, <img_w>, <rle>
        """
        outdir = os.path.join(self.outdir, self.datasrc)
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, self.video + ".txt")
        for t_idx, tracklet in enumerate(self.tracks):
            for time, id in enumerate(tracklet):
                if id == -1:
                    continue
                x1, y1, x2, y2 = self.all_proposals[id]['bbox']
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                conf = self.all_proposals[id][self.scoring]
                mask_rle = self.all_proposals[id]['instance_mask']
                img_h, img_w = mask_rle['size']
                rle_str = mask_rle['counts']
                with open(outpath, 'a') as f:
                    line = '%d,%d,%.2f,%.2f,%.2f,%.2f,%f,-1,-1,-1,%d,%d,%s\n' % (
                        time + 1, t_idx + 1, x1, y1, w, h, conf, img_h, img_w, rle_str)
                    f.write(line)

    def _preprocess_props(self, proposals, start_id):
        pid = start_id
        for prop in proposals:
            prop["id"] = pid
            pid += 1
            if self.dataset_name == "TAO":
                prop["embeddings"] = np.array(prop["embeddings"]) / 1e4
        return proposals

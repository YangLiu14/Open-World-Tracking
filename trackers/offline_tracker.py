import os

import numpy as np

from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

from base_tracker import BaseTracker
from online_tracker import OnlineTracker, OnlineGhostTracker
from tools.logger import create_log


class OfflineTracker(BaseTracker):
    def __init__(self, config):
        BaseTracker.__init__(self, config)
        self.pre_merge = config["tracker"]["pre_merge"]
        if self.pre_merge == "online":
            self.online_tracker = OnlineTracker(config)
        elif self.pre_merge == "online-ghost":
            self.online_tracker = OnlineGhostTracker(config)
        else:
            raise Exception(f"Setting online tracker for offline merge, "
                            f"but {self.pre_merge} is not implemented.")
        self.logger = create_log()
        self.logger.info("Logging is launched...")

    @property
    def outdir(self):
        return os.path.join(self.outroot, "offline", self.scoring)

    def _run(self, datasrc, video):
        self.datasrc = datasrc
        self.video = video
        self.tracklets = list()
        self.all_proposals = dict()

        self.tracklets, proposals_per_video, self.all_proposals = self.online_tracker._run(datasrc, video)

        all_tracklet_proposals = list(proposals_per_video.values())

        tracklet_seqs, trajectories = [], []
        trajectories_timeline = []

        tracklets_info = self.get_tracklets_info(all_tracklet_proposals)

        if len(self.tracklets) > 0:

            timeline = np.arange(len(self.tracklets[0]))

            sorted_tracklets_info = sorted(tracklets_info, key=lambda i: (i['start'], i['end']))
            norm_distance = self.normalization_distance(sorted_tracklets_info)
            for t in timeline:
                start_tt_tracklet = list(filter(lambda sort_track: sort_track['start'] == t, sorted_tracklets_info))
                end_before_tt_tracklet = list(filter(lambda sort_track: sort_track['end'] < t, sorted_tracklets_info))
                if start_tt_tracklet and end_before_tt_tracklet:
                    for ts in start_tt_tracklet:
                        self.merge_compability(ts, end_before_tt_tracklet, norm_distance)

            isVisited = np.full(len(tracklets_info), False, dtype=bool)

            for i, t in enumerate(tracklets_info[::-1]):
                if not isVisited[t['tracklet_id']]:
                    sequence = []
                    self.merge_tracklet(tracklets_info, t['tracklet_id'], sequence, isVisited)
                    tracklet_seqs.append(sequence)

            if len(tracklet_seqs) > 0:
                self.generate_refined_clusters(tracklet_seqs, tracklets_info, trajectories,
                                               time=timeline.shape[0], norm_distance=norm_distance)
            else:
                self.logger.info('Tracklets could not been merged...')

            for i, trj in enumerate(trajectories):
                seq_timeline = np.negative(np.ones_like(timeline, dtype=int))

                for t_id in trj:
                    ss = tracklets_info[t_id]['start']
                    ee = tracklets_info[t_id]['end'] + 1
                    seq_timeline[ss:ee] = self.tracklets[t_id][ss:ee]
                trajectories_timeline.append(seq_timeline)

        else:
            self.logger.info('Tracklets list is empty...')

        self.tracks = trajectories_timeline

    def calculate_merge_score(self, tracklet, merge_props, norm_distance, time_weighting=False,
                              alpha=0.6, distance_metric="L2"):
        if distance_metric == "L2":
            if time_weighting:  # Average ReID L2 distance with time weighting
                score = np.array(
                    [norm(tracklet['avg_embedding'] - prop['avg_embedding']) + (
                                alpha * abs(tracklet['start'] - prop['end']))
                     for prop in merge_props])
            else:  # Average ReID L2 distance
                score = np.array([norm(tracklet['avg_embedding'] - prop['avg_embedding']) / norm_distance
                                  for prop in merge_props])

                if any(score > 1) or any(score < 0):
                    self.logger.info("There is a problem while normalizing embeddings.")
            return [prop for _, prop in sorted(zip(score, merge_props), key=lambda t: t[0])]

        elif distance_metric == "cosine":
            if time_weighting:  # Average ReID L2 distance with time weighting
                raise Exception("Cosine based distance time_weighting is not implemented")
            else:  # Average ReID L2 distance
                score = np.array([cosine_similarity(tracklet['avg_embedding'].reshape(1, -1),
                                               prop['avg_embedding'].reshape(1, -1)) for prop in merge_props])

                if any(score > 1) or any(score < 0):
                    self.logger.info("There is a problem while normalizing embeddings.")
            return [prop for _, prop in sorted(zip(score, merge_props), key=lambda t: t[0], reverse=True)]

        else:
            raise Exception("Invalid distance metric")

    def cluster_tracklet_sequences(self, tracklet_seqs):
        clustered_tracklet_seqs = []
        for ts in tracklet_seqs:
            clustered = False
            if not clustered_tracklet_seqs:
                clustered_tracklet_seqs.append({'cluster_id': ts[0], 'clustered_sequences': [], 'refined': False})
                clustered_tracklet_seqs[0]['clustered_sequences'].append(ts)
                continue
            for cluster in clustered_tracklet_seqs:
                if cluster['cluster_id'] == ts[0]:
                    cluster['clustered_sequences'].append(ts)
                    clustered = True
                    break
            if not clustered:
                clustered_tracklet_seqs.append({'cluster_id': ts[0], 'clustered_sequences': [], 'refined': False})
                clustered_tracklet_seqs[-1]['clustered_sequences'].append(ts)
        return clustered_tracklet_seqs

    def generate_refined_clusters(self, tracklet_seqs, tracklets_info, trajectories, time, norm_distance):
        if len(tracklet_seqs) == 0:
            return

        tracklet_seqs = sorted([sorted(ts) for ts in tracklet_seqs])
        clustered_tracklet_seqs = self.cluster_tracklet_sequences(tracklet_seqs)

        for clustered_seqs in clustered_tracklet_seqs:
            if clustered_seqs['refined'] == False:
                cluster = clustered_seqs['clustered_sequences']
                trj, cluster = self.get_best_trajectory_within_cluster(cluster, tracklets_info, time,
                                                                  norm_distance=norm_distance)
                trajectories.append(trj)
                refined_cluster = self.refine_cluster(cluster, trj)
                self.generate_refined_clusters(refined_cluster, tracklets_info, trajectories, time=time,
                                          norm_distance=norm_distance)
                clustered_seqs['refined'] = True

    def get_best_trajectory_within_cluster(self, clustered_seqs, tracklets_info, time, norm_distance):
        ends = []
        covers = []
        visual_similarity = []
        for tracklet_seq in clustered_seqs:
            total_gap = 0
            distance = []
            for i, t_id in enumerate(tracklet_seq):
                total_gap += tracklets_info[t_id]['end'] - tracklets_info[t_id]['start']
                distance += [
                    1 - (norm(
                        tracklets_info[t_id]['avg_embedding'] - tracklets_info[t]['avg_embedding']) / norm_distance)
                    for t in tracklet_seq if t > t_id]
            covers.append(total_gap)
            if distance:
                visual_similarity.append(max(distance))
            else:
                visual_similarity.append(0)
            ends.append(tracklets_info[tracklet_seq[-1]]['end'])
        covers = np.array(covers) / time
        visual_similarity = np.array(visual_similarity)
        score = 0.9 * covers + 0.1 * visual_similarity

        if np.isnan(score).any() == True:
            nan_indices = np.where(np.isnan(score) == True)[0]
            for i in nan_indices:
                score[i] = 0

        max_score = np.max(score)
        best_ind = np.where(score == max_score)[0]
        if len(best_ind) > 1:
            ind = best_ind[np.argmax([ends[i] for i in best_ind])]
        else:
            ind = best_ind[0]

        sequence = clustered_seqs[ind]
        clustered_seqs.pop(ind)

        return sequence, clustered_seqs

    def get_tracklets_info(self, all_tracklet_prop):
        tracklets_info = self.get_tracklets_temporal_info()

        for i, ts in enumerate(self.tracklets):
            # print("Info Tracklet in {0}/{1}".format(i + 1, len(tracklets)))
            t_start = tracklets_info[i]['start']
            t_end = tracklets_info[i]['end']

            t_reid = np.array([all_tracklet_prop[t][ind]['embeddings'] for t in range(t_start, t_end + 1) for ind in
                          range(len(all_tracklet_prop[t])) if all_tracklet_prop[t][ind]['id'] == ts[t]])

            t_avg_reid = np.mean(t_reid, axis=0)
            tracklets_info[i]['avg_embedding'] = t_avg_reid

        return tracklets_info

    def get_tracklets_temporal_info(self):
        tracklets_info = []
        for i, ts in enumerate(self.tracklets):
            t_start = np.where(ts != -1)[0][0]
            r = np.where(ts == -1)[0]
            if len(r[r > t_start]) > 0:
                # if the tracklet sequence runs at the end of time line,
                # then seq_end is null, in that case seq_end is the last index
                t_end = r[r > t_start][0] - 1
            else:
                t_end = len(ts) - 1
            tracklets_info.append({'tracklet_id': i, 'start': t_start, 'end': t_end, 'backward_merged_prop': -1})

        return tracklets_info

    def merge_compability(self, ts, end_before_tt_tracklet, norm_distance):
        merge_prop = self.calculate_merge_score(ts, end_before_tt_tracklet, norm_distance=norm_distance)
        if len(merge_prop) >= 2:
            first_best = merge_prop[0]
            second_best = merge_prop[1]
            if first_best['tracklet_id'] == second_best['backward_merged_prop']:
                potential_pred = second_best
                new_end_before_tt = [tt for tt in end_before_tt_tracklet if
                                     tt['start'] < potential_pred['end'] and tt['backward_merged_prop'] ==
                                     potential_pred[
                                         'tracklet_id']]
                new_end_before_tt.append(potential_pred)
                self.merge_compability(ts, new_end_before_tt, norm_distance)
            else:
                ts['backward_merged_prop'] = first_best['tracklet_id']
        else:
            ts['backward_merged_prop'] = merge_prop[0]['tracklet_id']

    def merge_tracklet(self, tracklets, tracklet_id, sequence, isVisited):
        sequence.append(tracklet_id)
        merge_id = tracklets[tracklet_id]['backward_merged_prop']

        if merge_id == -1:
            isVisited[tracklet_id] = True
            return

        isVisited[tracklet_id] = True
        self.merge_tracklet(tracklets, merge_id, sequence, isVisited)

    def normalization_distance(self, tracklets_info):
        distances = []
        for i, trk in enumerate(tracklets_info):
            for j in range(i + 1, len(tracklets_info)):
                distances.append(norm(trk['avg_embedding'] - tracklets_info[j]['avg_embedding']))

        return max(distances)

    def refine_cluster(self, clustered_seqs, sequence):
        for c_seq in clustered_seqs:
            for t_id in sequence:
                if t_id not in c_seq:
                    break
                else:
                    c_seq.remove(t_id)

        return clustered_seqs



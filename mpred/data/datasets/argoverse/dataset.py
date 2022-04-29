from mai.utils import FI
from mai.utils import io
import numpy as np
import os

from torch.distributed import get_rank

from mpred.data.datasets.mpred_dataset import MPredDataset
from mpred.core.annotation_pred import AnnotationTrajPred

from argoverse.evaluation.competition_util import generate_forecasting_h5
from argoverse.map_representation.map_api import ArgoverseMap


@FI.register
class ArgoPredDataset(MPredDataset):
    def __init__(self, info_path, load_opt={}, filters=[], transforms=[], codec=None):
        super().__init__(info_path, load_opt, filters, transforms, codec)
        self.load_opt = load_opt
        self.lane_dict, self.lane_id2idx = io.load(load_opt['map_path'])
        self.trajs = np.load(load_opt['traj_path'])
        self.am = ArgoverseMap()

    def load_meta(self, sample, info):
        sample['meta'] = dict(sample_id=info['id'],
                              city=info['city'],
                              obs_len=self.load_opt['obs_len'],
                              pred_len=self.load_opt['pred_len'],
                              lane_radius=self.load_opt['lane_radius'],
                              target_id=0,
                              )

    def load_data(self, sample, info):
        traj_index = info['traj_index']
        agent = self.trajs[traj_index[0]:traj_index[1]].copy()

        city = info['city']
        agent_pos = agent[0, self.load_opt['obs_len']-1, :2]

        lane_ids = self.am.get_lane_ids_in_xy_bbox(
            agent_pos[0], agent_pos[1], city, self.load_opt['lane_radius'])
        lanes = []
        for id in lane_ids:
            indices = self.lane_id2idx[city][id]
            lanes.extend([self.lane_dict[city][idx] for idx in indices])
        lanes = np.stack(lanes, axis=0)

        sample['data'] = dict(agent=agent, lane=lanes)

    def load_anno(self, sample, info):
        agent_idx = info['traj_index'][0]
        pred_traj = self.trajs[[agent_idx],
                               self.load_opt['obs_len']:, :2].copy()
        sample['anno'] = AnnotationTrajPred(trajs=pred_traj)

    @classmethod
    def format(cls, sample_list, pred_path=None, gt_path=None):
        if not (pred_path is None and gt_path is None):
            meta_list = [sample['meta'] for sample in sample_list]

        # process prediction
        if pred_path is not None:
            print('Formatting predictions...')
            pred_list = [sample['pred'] for sample in sample_list]
            pred_pb = cls._format_anno_list(pred_list, meta_list)
            print(f'Saving formatted predictions into {pred_path}')
            io.dump(pred_pb, pred_path, format='pkl')
            #  generate_forecasting_h5(
            #      data=pred_pb['trajs'],
            #      output_path='/tmp',
            #      filename=f'pred.{get_rank()}',
            #      probabilities=pred_pb['scores'])

        # process anno
        if gt_path is not None:
            print('Formatting groundtruth...')
            gt_list = [sample['anno'] for sample in sample_list]
            gt_pb = cls._format_anno_list(gt_list, meta_list)
            print(f'Saving formatted groundtruth into {gt_path}')
            io.dump(gt_pb, gt_path, format='pkl')

        return pred_path, gt_path

    @classmethod
    def _format_anno_list(cls, anno_list, meta_list):
        from mai.core.geometry2d import rotate_points
        trajs = {}
        scores = {}
        for anno, meta in zip(anno_list, meta_list):
            id = meta['sample_id']

            traj = anno.trajs
            score = anno.scores

            if 'norm_center' in meta:
                norm_center = meta['norm_center']
                norm_rot = meta['norm_rot']
                norm_rot_inv = np.array([norm_rot[0], -norm_rot[1]])

                traj = rotate_points(anno.trajs, norm_rot_inv) + norm_center

            if len(traj) > 1:  # pred
                trajs[id] = list(traj)
            else:  # gt
                trajs[id] = traj[0]

            scores[id] = list(score)

        return dict(trajs=trajs, scores=scores)

    @classmethod
    def evaluate(cls, predict_path, gt_path=None):
        from .utils import compute_forecasting_metrics

        pred = io.load(predict_path, format='pkl')
        gt = io.load(gt_path, format='pkl')

        metric = compute_forecasting_metrics(
            forecasted_trajectories=pred['trajs'],
            gt_trajectories=gt['trajs'],
            max_n_guesses=6,
            horizon=30,
            miss_threshold=2.0,
            forecasted_probabilities=pred['scores'],
        )

        return metric

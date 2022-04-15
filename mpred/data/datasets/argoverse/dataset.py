from mai.utils import FI
from mai.utils import io
import numpy as np

from mpred.data.datasets.mpred_dataset import MPredDataset
from mpred.core.annotation_pred import AnnotationTrajPred


@FI.register
class ArgoPredDataset(MPredDataset):
    def __init__(self, info_path, load_opt={}, filters=[], transforms=[], codec=None):
        super().__init__(info_path, filters, transforms, codec)
        self.load_opt = load_opt
        self.lane_dict, self.lane_id2idx = io.load(load_opt['map_path'])

    def load_meta(self, sample, info):
        sample['meta'] = dict(sample_id=info['id'], city=info['city'],
                              obs_len=self.load_opt['obs_len'], pred_len=self.load_opt['pred_len'])

    def load_data(self, sample, info):
        agent = info['agent'][:, :self.load_opt['obs_len']]

        # filter object that has no data in 0:obs_len
        mask = np.any(agent[:, :, -1] > 0, axis=1)
        agent = agent[mask]

        city = info['city']
        lane_ids = info['lane']
        lane = []
        for id in lane_ids:
            indices = self.lane_id2idx[city][id]
            lane.extend([self.lane_dict[city][idx] for idx in indices])
        lane = np.stack(lane, axis=0)

        sample['data'] = dict(agent=agent, lane=lane)

    def load_anno(self, sample, info):
        pred_traj = info['agent'][0,
                                  self.load_opt['obs_len']:, :2][np.newaxis, :]
        sample['anno'] = AnnotationTrajPred(trajs=pred_traj)

    def format(self, sample_list, pred_path=None, gt_path=None):
        if not (pred_path is None and gt_path is None):
            meta_list = [sample['meta'] for sample in sample_list]

        # process prediction
        if pred_path is not None:
            print('Formatting predictions...')
            pred_list = [sample['pred'] for sample in sample_list]
            pred_pb = self._format_anno_list(pred_list, meta_list)
            io.dump(pred_pb, pred_path, format='pkl')
            print(f'Save formatted predictions into {pred_path}')

        # process anno
        if gt_path is not None:
            print('Formatting groundtruth...')
            gt_list = [sample['anno'] for sample in sample_list]
            gt_pb = self._format_anno_list(gt_list, meta_list)
            io.dump(gt_pb, gt_path, format='pkl')
            print(f'Save formatted groundtruth into {gt_path}')

        return pred_path, gt_path

    def _format_anno_list(self, anno_list, meta_list):
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

    def evaluate(self, predict_path, gt_path=None):
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

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

        city = info['city']
        lane = info['lane']
        lane = [self.lane_dict[city][self.lane_id2idx[city][id]]
                for id in lane]
        lane = np.stack(lane, axis=0)

        sample['data'] = dict(agent=agent, lane=lane)

    def load_anno(self, sample, info):
        pred_traj = info['agent'][0, self.load_opt['obs_len']:][np.newaxis, :]
        sample['anno'] = AnnotationTrajPred(trajs=pred_traj)

    def format(self, result, pred_path=None, gt_path=None):
        # TODO
        raise NotImplementedError

    def evaluate(self, predict_path, gt_path=None):
        # TODO
        raise NotImplementedError

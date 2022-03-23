from mai.utils import FI
from mai.data.datasets.transform import DatasetTransform
import numpy as np


@FI.register
class Normlize(DatasetTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, sample, info):
        obs_len = sample['meta']['obs_len']
        agent = sample['data']['agent']
        lane = sample['data']['lane']
        anno = sample['anno']

        agent_pos = np.array(agent[0, obs_len-1, :2], dtype=np.float32)

        agent[:, :, :2] -= agent_pos
        lane[:, :, :2] -= agent_pos
        anno.trajs[:, :, :2] -= agent_pos

        sample['meta']['norm_center'] = agent_pos

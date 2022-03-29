from mai.utils import FI
from mai.data.datasets.transform import DatasetTransform
import numpy as np
from mdet.core.geometry2d import rotate_points


@FI.register
class Normlize(DatasetTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, sample, info):
        obs_len = sample['meta']['obs_len']
        agent = sample['data']['agent']
        lane = sample['data']['lane']
        anno = sample['anno']

        agent_pos = agent[0, obs_len-1, :2].copy()
        agent_rot = agent[0, obs_len-1, :2] - agent[0, obs_len-2, :2]
        agent_rot = agent_rot / (np.linalg.norm(agent_rot) + 1e-6)
        agent_rot[1] = -agent_rot[1] 

        agent[:, :, :2] = rotate_points(agent[:, :, :2] - agent_pos, agent_rot)
        lane[:, :, :2] = rotate_points(lane[:, :, :2]-agent_pos, agent_rot)
        anno.trajs[:, :, :2] = rotate_points(
            anno.trajs[:, :, :2]-agent_pos, agent_rot)

        sample['meta']['norm_center'] = agent_pos
        sample['meta']['norm_rot'] = agent_rot

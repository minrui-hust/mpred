from mai.utils import FI
from mai.data.datasets.transform import DatasetTransform
import numpy as np
from mdet.core.geometry2d import rotate_points


@FI.register
class Normlize(DatasetTransform):
    def __init__(self, center=True, orient=True):
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


@FI.register
class ObjectRangeFilter(DatasetTransform):
    def __init__(self, obj_radius=50):
        super().__init__()
        self.obj_radius = obj_radius

    def __call__(self, sample, info):
        agent = sample['data']['agent']
        agent_pos = agent[0, -1, :2].copy()

        object_dist = np.linalg.norm(agent[:, -1, :2] - agent_pos, axis=-1)
        valid_mask = object_dist <= self.obj_radius

        sample['data']['agent'] = agent[valid_mask]


@FI.register
class MpredMirrorFlip(DatasetTransform):
    def __init__(self, mirror_prob=0.5, flip_prob=0.5):
        super().__init__()

        self.mirror_prob = mirror_prob
        self.flip_prob = flip_prob

    def __call__(self, sample, info):
        self._mirror(sample)
        self._flip(sample)

    def _mirror(self, sample):
        if self.mirror_prob < np.random.rand():
            return

        agent = sample['data']['agent']
        lane = sample['data']['lane']
        trajs = sample['anno'].trajs

        agent[:, :, 0] = -agent[:, :, 0]
        lane[:, :, 0] = -lane[:, :, 0]
        trajs[:, :, 0] = -trajs[:, :, 0]

        sample['data']['agent'] = agent
        sample['data']['lane'] = lane
        sample['anno'].trajs = trajs

    def _flip(self, sample):
        if self.flip_prob < np.random.rand():
            return

        agent = sample['data']['agent']
        lane = sample['data']['lane']
        trajs = sample['anno'].trajs

        agent[:, :, 1] = -agent[:, :, 1]
        lane[:, :, 1] = -lane[:, :, 1]
        trajs[:, :, 1] = -trajs[:, :, 1]

        sample['data']['agent'] = agent
        sample['data']['lane'] = lane
        sample['anno'].trajs = trajs


@FI.register
class MpredGlobalTransform(DatasetTransform):
    def __init__(self, translation_std=[0, 0], rot_range=[-0.78539816, 0.78539816], scale_range=[0.95, 1.05]):
        super().__init__()

        self.translation_std = np.array(translation_std, dtype=np.float32)
        self.rot_range = rot_range
        self.scale_range = scale_range

    def __call__(self, sample, info):
        self._retarget(sample)
        self._scale(sample)
        self._rotate(sample)
        self._translate(sample)

    def _retarget(self, sample):
        r'''
        random select a object as target coordinate frame
        '''
        agent = sample['data']['agent']
        target_pos = agent[np.random.randint(len(agent))][-1, :2].copy()

        sample['data']['agent'][:, :, :2] -= target_pos
        sample['data']['lane'][:, :, :2] -= target_pos
        sample['anno'].trajs[:, :, :2] -= target_pos

    def _scale(self, sample):
        scale_factor = np.random.uniform(
            self.scale_range[0], self.scale_range[1])

        sample['data']['agent'][:, :, :2] *= scale_factor
        sample['data']['lane'][:, :, :2] *= scale_factor
        sample['anno'].trajs[:, :, :2] *= scale_factor

    def _rotate(self, sample):
        alpha = np.random.uniform(self.rot_range[0], self.rot_range[1])
        rot = np.array([np.cos(alpha), np.sin(alpha)], dtype=np.float32)

        agent = sample['data']['agent']
        lane = sample['data']['lane']
        trajs = sample['anno'].trajs

        agent[:, :, :2] = rotate_points(agent[:, :, :2], rot)
        lane[:, :, :2] = rotate_points(lane[:, :, :2], rot)
        trajs[:, :, :2] = rotate_points(trajs[:, :, :2], rot)

    def _translate(self, sample):
        translation = np.random.normal(scale=self.translation_std, size=2)

        sample['data']['agent'][:, :, :2] -= translation
        sample['data']['lane'][:, :, :2] -= translation
        sample['anno'].trajs[:, :, :2] -= translation

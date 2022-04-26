from mai.utils import FI
from mai.data.datasets.transform import DatasetTransform
import numpy as np
from mdet.core.geometry2d import rotate_points


@FI.register
class MpredRetarget(DatasetTransform):
    r'''
    select another object as prediction target
    '''

    def __init__(self, prob=0.0, min_len=40, min_dist=0.2):
        super().__init__()
        self.prob = prob
        self.min_len = min_len
        self.min_dist = min_dist

    def __call__(self, sample, info, ds=None):
        if self.prob < np.random.rand():
            return

        city = sample['meta']['city']
        obs_len = sample['meta']['obs_len']
        pred_len = sample['meta']['pred_len']
        lane_radius = sample['meta']['lane_radius']
        total_len = obs_len + pred_len

        traj = sample['data']['agent'][1:]
        traj_valid_len = np.sum(traj[..., -1], axis=-1).astype(np.int32)
        traj_dist = np.linalg.norm(
            traj[:, obs_len-1, :2]-traj[:, 0, :2], axis=-1)

        traj_len = []
        for m in traj[..., -1]:
            indice = np.nonzero(m)[0]
            st, ed = indice[0], indice[-1]
            traj_len.append(ed-st+1)
        traj_len = np.array(traj_len, dtype=np.int32)

        mask = np.logical_and(traj_valid_len >= self.min_len,
                              traj_dist >= self.min_dist)
        mask = np.logical_and(mask, traj_valid_len == traj_len)
        candi_indice = np.nonzero(mask)[0]
        if len(candi_indice) <= 0:
            return

        sel = np.random.randint(len(candi_indice))
        candi = candi_indice[sel]
        candi_len = traj_len[candi]
        st = traj[candi, :, -1].argmax()
        ed = st+candi_len
        candi_pos = traj[candi, ed-pred_len, :2]

        lane_ids = ds.am.get_lane_ids_in_xy_bbox(
            candi_pos[0], candi_pos[1], city, lane_radius)
        if len(lane_ids) <= 0:
            return

        # swap selected target to be the first agent
        agent = sample['data']['agent']
        ts = agent[..., 2].copy()
        agent[[0, candi+1]] = agent[[candi+1, 0]]

        pad = total_len-ed
        if pad > 0:
            agent = agent[:, :ed, :]
            agent = np.pad(agent, ((0, 0), (pad, 0), (0, 0)), mode='edge')
            agent[..., :pad, -1] = 0
            agent[..., 2] = ts

        lanes = []
        for id in lane_ids:
            indices = ds.lane_id2idx[city][id]
            lanes.extend([ds.lane_dict[city][idx] for idx in indices])
        lanes = np.stack(lanes, axis=0)

        gt_traj = agent[:1, obs_len:, :2].copy()

        sample['data']['agent'] = agent
        sample['data']['lane'] = lanes
        sample['anno'].trajs = gt_traj
        sample['meta']['target_id'] = candi+1


@FI.register
class Normalize(DatasetTransform):
    def __init__(self, center=True, orient=True):
        super().__init__()

    def __call__(self, sample, info, ds=None):
        obs_len = sample['meta']['obs_len']
        agent = sample['data']['agent']
        lane = sample['data']['lane']

        agent_pos = agent[0, obs_len-1, :2].copy()

        valid_last_id = obs_len-2
        agent_rot = np.array([1, 0], dtype=np.float32)
        while valid_last_id >= 0:
            agent_pos_last = agent[0, valid_last_id, :2]
            delta = agent_pos - agent_pos_last
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 1e-6:
                agent_rot = delta / delta_norm
                break
            valid_last_id -= 1

        agent_rot[1] = -agent_rot[1]

        agent[:, :, :2] = rotate_points(agent[:, :, :2] - agent_pos, agent_rot)
        lane[:, :, :2] = rotate_points(lane[:, :, :2]-agent_pos, agent_rot)

        if 'anno' in sample:
            anno = sample['anno']
            anno.trajs = rotate_points(anno.trajs-agent_pos, agent_rot)

        sample['meta']['norm_center'] = agent_pos
        sample['meta']['norm_rot'] = agent_rot


@FI.register
class MpredMaskHistory(DatasetTransform):
    def __init__(self, mask_prob=0.5, max_len=10):
        self.mask_prob = mask_prob
        self.max_len = max_len

    def __call__(self, sample, info, ds=None):
        agent = sample['data']['agent']
        mask = np.random.rand(agent.shape[0]) < self.mask_prob
        mask_len = np.random.randint(self.max_len+1, size=agent.shape[0])

        indices = np.nonzero(mask)[0]
        for i in indices:
            agent[i, :mask_len[i], -1] = 0
            agent[i, :mask_len[i], :2] = agent[i, mask_len[i], :2]


@FI.register
class ObjectRangeFilter(DatasetTransform):
    def __init__(self, obj_radius=50):
        super().__init__()
        self.obj_radius = obj_radius

    def __call__(self, sample, info, ds=None):
        obs_len = sample['meta']['obs_len']

        agent = sample['data']['agent']
        agent_pos = agent[:, obs_len-1, :2]

        object_dist = np.linalg.norm(agent_pos - agent_pos[0], axis=-1)
        valid_mask = object_dist <= self.obj_radius

        sample['data']['agent'] = agent[valid_mask]


@FI.register
class MpredMirrorFlip(DatasetTransform):
    def __init__(self, mirror_prob=None, flip_prob=None):
        super().__init__()

        self.mirror_prob = mirror_prob
        self.flip_prob = flip_prob

    def __call__(self, sample, info, ds=None):
        if self.mirror_prob is not None:
            self._mirror(sample)
        if self.flip_prob is not None:
            self._flip(sample)

    def _mirror(self, sample):
        if self.mirror_prob < np.random.rand():
            return

        agent = sample['data']['agent']
        lane = sample['data']['lane']

        sample['data']['agent'][..., 0] = -agent[..., 0]
        sample['data']['lane'][..., 0] = -lane[..., 0]

        if 'anno' in sample:
            sample['anno'].trajs[..., 0] = -sample['anno'].trajs[..., 0]

    def _flip(self, sample):
        if self.flip_prob < np.random.rand():
            return

        agent = sample['data']['agent']
        lane = sample['data']['lane']

        sample['data']['agent'][..., 1] = -agent[..., 1]
        sample['data']['lane'][..., 1] = -lane[..., 1]

        if 'anno' in sample:
            sample['anno'].trajs[..., 1] = -sample['anno'].trajs[..., 1]


@FI.register
class MpredGlobalTransform(DatasetTransform):
    def __init__(self, translation_std=None, rot_range=None, scale_range=None):
        super().__init__()

        self.translation_std = None
        if translation_std is not None:
            self.translation_std = np.array(translation_std, dtype=np.float32)
        self.rot_range = rot_range
        self.scale_range = scale_range

    def __call__(self, sample, info, ds=None):
        if self.scale_range is not None:
            self._scale(sample)
        if self.rot_range is not None:
            self._rotate(sample)
        if self.translation_std is not None:
            self._translate(sample)

    def _scale(self, sample):
        scale_factor = np.random.uniform(
            self.scale_range[0], self.scale_range[1])

        sample['data']['agent'][..., :2] *= scale_factor
        sample['data']['lane'][..., :2] *= scale_factor

        if 'anno' in sample:
            sample['anno'].trajs[..., :2] *= scale_factor

    def _rotate(self, sample):
        alpha = np.random.uniform(self.rot_range[0], self.rot_range[1])
        rot = np.array([np.cos(alpha), np.sin(alpha)], dtype=np.float32)

        agent = sample['data']['agent']
        lane = sample['data']['lane']

        agent[..., :2] = rotate_points(agent[..., :2], rot)
        lane[..., :2] = rotate_points(lane[..., :2], rot)

        if 'anno' in sample:
            trajs = sample['anno'].trajs
            trajs[..., :2] = rotate_points(trajs[..., :2], rot)

    def _translate(self, sample):
        translation = np.random.normal(scale=self.translation_std, size=2)

        sample['data']['agent'][..., :2] -= translation
        sample['data']['lane'][..., :2] -= translation
        if 'anno' in sample:
            sample['anno'].trajs[..., :2] -= translation

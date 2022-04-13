from mai.data.codecs import BaseCodec
from mai.data.collators import Collator
from mai.utils import FI
import numpy as np
import torch
import torch.nn.functional as F
from mpred.core.annotation_pred import AnnotationTrajPred


@FI.register
class MMTransCodec(BaseCodec):
    def __init__(self, encode_cfg={}, decode_cfg={}, loss_cfg={}):
        super().__init__(encode_cfg, decode_cfg, loss_cfg)

        self.kl_loss = torch.nn.KLDivLoss(
            log_target=True, reduction='batchmean')

    def encode_data(self, sample, info):
        # encode agent
        agent = sample['data']['agent']
        mask = agent[:, :, -1]
        last_idx = []
        for m in mask:
            last_idx.append(np.nonzero(m)[0].max())
        last_idx = np.array(last_idx, dtype=np.int32)

        pos = agent[:, :, :2]
        delta = pos[:, :-1] - pos[:, 1:]
        agent = np.concatenate([delta, agent[:, :-1, 2:]], axis=-1)
        pos = pos[:, -1]
        agent_num = np.array(len(agent), dtype=np.int32)

        # encode lane
        lane = sample['data']['lane']
        lane = np.concatenate(
            [lane[:, :-1, :2], lane[:, 1:, :]], axis=-1)  # vectorize
        lane_num = np.array(len(lane), dtype=np.int32)

        sample['input'] = dict(
            agent=torch.from_numpy(agent),
            pos=torch.from_numpy(pos),
            lane=torch.from_numpy(lane),
            agent_num=torch.from_numpy(agent_num),
            lane_num=torch.from_numpy(lane_num),
        )

        sample['meta']['last_idx'] = last_idx

    def encode_anno(self, sample, info):
        traj = sample['anno'].trajs[0]
        sample['gt'] = dict(
            traj=torch.from_numpy(traj),
        )

    def decode_eval(self, output, batch=None):
        r'''
        output --> pred
        '''

        L = batch['meta'][0]['pred_len']

        traj = output['traj'][:, 0]
        traj = traj.reshape(*traj.shape[:-1], L, 2).cpu()  # B, K, L, 2

        score = torch.softmax(output['score'][:, 0].squeeze(-1), dim=1).cpu()

        pred_list = []
        for i in range(len(traj)):
            pred = AnnotationTrajPred(
                trajs=traj[i].numpy(), scores=score[i].numpy())
            pred_list.append(pred)

        return pred_list

    def decode_export(self, output, batch=None):
        L = 30

        traj = output['traj'][:, 0]
        traj = traj.reshape(*traj.shape[:-1], L, 2)  # (B, K, L, 2)

        score = torch.softmax(
            output['score'][:, 0].squeeze(-1), dim=1)  # (B, K)

        return traj, score

    def decode_trt(self, output, batch=None):
        pass

    def loss(self, output, batch):
        pred_traj = output['traj'][:, 0]
        (B, K), L = pred_traj.shape[:2], int(pred_traj.shape[2]/2)

        pred_traj = pred_traj.view(B, K, L, 2)

        gt_traj = batch['gt']['traj'].unsqueeze(1)  # B, 1, L, 2

        traj_dist = torch.linalg.norm(
            pred_traj[:, :, -1, :] - gt_traj[:, :, -1, :], dim=-1)
        min_idx = torch.min(traj_dist, dim=-1)[1]

        pred_traj = torch.gather(pred_traj, 1, min_idx.view(
            min_idx.shape[0], 1, 1, 1).expand(-1, -1, pred_traj.shape[2], pred_traj.shape[3]))

        traj_loss = F.huber_loss(
            pred_traj, gt_traj, delta=self.loss_cfg['delta'])

        gt_score = F.one_hot(min_idx, K).float()
        pred_score = F.sigmoid(output['score'][:, 0].squeeze(-1))  # B, K
        score_loss = F.cross_entropy(pred_score, gt_score)

        loss = self.loss_cfg['wgt']['traj'] * traj_loss + \
            self.loss_cfg['wgt']['score'] * score_loss

        return dict(loss=loss, traj_loss=traj_loss, score_loss=score_loss)

    def collater(self):
        def pad_func(data, data_list):
            A = max([len(d) for d in data_list])
            a = len(data)
            pad = [0, 0] * data.dim()
            pad[-1] = A-a
            pad_cfg = dict(pad=tuple(pad), mode='constant', value=0)
            return pad_cfg

        rules = {
            # rules for data
            '.data': dict(type='append'),

            # rules for anno
            '.anno': dict(type='append'),

            # rules for input
            '.input.agent': dict(type='stack', pad_func=pad_func),
            '.input.pos': dict(type='stack', pad_func=pad_func),
            '.input.lane': dict(type='stack', pad_func=pad_func),
            '.input.agent_num': dict(type='stack'),
            '.input.lane_num': dict(type='stack'),

            # rules for gt
            '.gt.traj': dict(type='stack'),

            # rules for output, would be very simillar to gt
            '.output.traj': dict(type='stack'),
            '.output.score': dict(type='stack'),

            # rules for meta
            '.meta': dict(type='append'),
        }

        return Collator(rules=rules)

    def plot(self, sample,
             show_input=True,
             show_gt=True,
             show_output=False,
             ):
        import matplotlib.pyplot as plt

        plt.axes().set_aspect('equal')
        plt.grid()

        if show_input and 'input' in sample:
            lane = sample['input']['lane']
            for l in lane:
                s = l[:, :2]
                e = l[:, 2:4]
                d = e-s
                plt.quiver(s[:, 0], s[:, 1], d[:, 0], d[:, 1], units='xy', angles='xy', scale=1.0, scale_units='xy',
                           width=0.4, headwidth=1.8, headlength=1.0, headaxislength=1.5, color='gray', alpha=0.8)

            L = sample['meta']['obs_len']
            c = torch.arange(L) / (L-1)

            # plot obj
            cm = plt.cm.get_cmap('winter')
            object = sample['input']['agent'][1:]
            object_pos = sample['input']['pos'][1:]
            last_idx = sample['meta']['last_idx'][1:]
            for obj, pos, last in zip(object, object_pos, last_idx):
                d = obj[:, :2]
                d = torch.cat([d, pos.unsqueeze(0)], dim=0)
                # reverse cumsum
                p = d + torch.sum(d, dim=0, keepdims=True) - \
                    torch.cumsum(d, dim=0)

                m = obj[:, -1]
                m = F.pad(m, (0, 1), value=0)
                m[last] = 1
                mask = m > 0

                p = p[mask]

                plt.scatter(p[:, 0], p[:, 1], s=28, c=cm(c[mask]))

            # plot agent
            cm = plt.cm.get_cmap('autumn')
            agent = sample['input']['agent'][0]
            agent_pos = sample['input']['pos'][0]
            last_idx = sample['meta']['last_idx'][0]
            d = agent[:, :2]
            d = torch.cat([d, agent_pos.unsqueeze(0)], dim=0)
            # reverse cumsum
            p = d + torch.sum(d, dim=0, keepdims=True) - \
                torch.cumsum(d, dim=0)

            m = agent[:, -1]
            m = F.pad(m, (0, 1), value=0)
            m[last_idx] = 1
            mask = m > 0

            p = p[mask]

            plt.scatter(p[:, 0], p[:, 1], s=28, c=cm(c[mask]))

        if show_gt and 'gt' in sample:
            traj = sample['gt']['traj']
            cm = plt.cm.get_cmap('spring')
            L = sample['meta']['pred_len']
            c = torch.arange(L) / (L-1)
            plt.scatter(traj[:, 0], traj[:, 1], s=28, c=cm(c))

        if show_output and 'output' in sample:
            pass

        plt.show()

    def export_info(self, batch):
        agent = batch['input']['agent']  # (B, A, 19, 4)
        lane = batch['input']['lane']  # (B, L, 9, 7)
        pos = batch['input']['pos']  # (B, A, 2)
        agent_num = batch['input']['agent_num']  # (B)
        lane_num = batch['input']['lane_num']  # (B)

        input = (agent, lane, pos, agent_num, lane_num, )
        input_name = ['agent', 'lane', 'pos', 'agent_num', 'lane_num']
        output_name = ['traj', 'score']
        dynamic_axes = {}

        return input, input_name, output_name, dynamic_axes

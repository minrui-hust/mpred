from mai.data.datasets import BaseDataset
import numpy as np
from mai.utils import FI


@FI.register
class MPredDataset(BaseDataset):
    def __init__(self, info_path, load_opt={}, filters=[], transforms=[], codec=None):
        super().__init__(info_path, filters, transforms, codec)
        self.load_opt = load_opt

    def load(self, sample, info):
        if self.load_opt.get('load_meta', True):
            self.load_meta(sample, info)
        if self.load_opt.get('load_data', True):
            self.load_data(sample, info)
        if self.load_opt.get('load_anno', True):
            self.load_anno(sample, info)

    def load_meta(self, sample, info):
        raise NotImplementedError

    def load_data(self, sample, info):
        raise NotImplementedError

    def load_anno(self, sample, info):
        raise NotImplementedError

    @classmethod
    def plot(cls, sample, show_data=True, show_anno=True, show_pred=False):
        r'''
        plot standard prediction sample
        '''

        import matplotlib.pyplot as plt

        print(sample['meta'])

        plt.title(str(sample['meta']['sample_id']).zfill(6))
        plt.axes().set_aspect('equal')
        plt.grid()

        if show_data and 'data' in sample:
            # plot lane
            lane = sample['data']['lane']
            for l in lane:
                d = l[1:, :2]-l[:-1, :2]
                s = l[:-1, :2]
                plt.quiver(s[:, 0], s[:, 1], d[:, 0], d[:, 1], units='xy', angles='xy', scale=1.0, scale_units='xy',
                           width=0.4, headwidth=1.8, headlength=1.0, headaxislength=1.5, color='gray', alpha=0.8)

            L = sample['meta']['obs_len']
            c = np.arange(L) / (L-1)

            # plot obj
            cm = plt.cm.get_cmap('winter')
            object = sample['data']['agent'][1:, :L]
            for obj in object:
                mask = obj[:, -1] > 0
                obj = obj[mask]
                plt.scatter(obj[:, 0], obj[:, 1], s=4, c=cm(c[mask]))

            # plot agent
            cm = plt.cm.get_cmap('autumn')
            agent = sample['data']['agent'][0, :L]
            plt.scatter(agent[:, 0], agent[:, 1], s=4, c=cm(c))

        if show_pred and 'pred' in sample:
            L = sample['meta']['pred_len']
            c = np.arange(L) / (L-1)

            cm = plt.cm.get_cmap('spring')
            for traj, score in zip(sample['pred'].trajs, sample['pred'].scores):
                plt.plot(traj[:, 0], traj[:, 1], 'g', lw=1, mew=3)
                #  plt.text(traj[-1, 0], traj[-1, 1], str(score))

        if show_anno and 'anno' in sample:
            L = sample['meta']['pred_len']
            c = np.arange(L) / (L-1)

            cm = plt.cm.get_cmap('gray')
            traj = sample['anno'].trajs[0]
            plt.plot(traj[:, 0], traj[:, 1], 'r', lw=1, mew=3)

        plt.show()

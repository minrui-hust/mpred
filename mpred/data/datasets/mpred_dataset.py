from mai.data.datasets import BaseDataset
import numpy as np


class MPredDataset(BaseDataset):
    def __init__(self, info_path, filters=[], transforms=[], codec=None):
        super().__init__(info_path, filters, transforms, codec)

    def load(self, sample, info):
        self.load_meta(sample, info)
        self.load_data(sample, info)
        self.load_anno(sample, info)

    def load_meta(self, sample, info):
        raise NotImplementedError

    def load_data(self, sample, info):
        raise NotImplementedError

    def load_anno(self, sample, info):
        raise NotImplementedError

    def plot(self, sample, show_data=True, show_anno=True, show_pred=False):
        r'''
        plot standard prediction sample
        '''

        import matplotlib.pyplot as plt

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
            object = sample['data']['agent'][1:]
            for obj in object:
                mask = obj[:, -1] > 0
                obj = obj[mask]
                plt.scatter(obj[:, 0], obj[:, 1], s=28, c=cm(c[mask]))

            # plot agent
            cm = plt.cm.get_cmap('autumn')
            agent = sample['data']['agent'][0]
            plt.scatter(agent[:, 0], agent[:, 1], s=32, c=cm(c))

        if show_anno and 'anno' in sample:
            L = sample['meta']['pred_len']
            c = np.arange(L) / (L-1)

            cm = plt.cm.get_cmap('spring')
            traj = sample['anno'].trajs[0]
            plt.scatter(traj[:, 0], traj[:, 1], s=32, c=cm(c), marker='H')

        if show_pred and 'pred' in sample:
            # plot pred traj
            pass

        plt.show()

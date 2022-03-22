from mpred.data.datasets.mpred_dataset import MPredDataset


class ArgoPredDataset(MPredDataset):
    def __init__(self, info_path, filters=[], transforms=[], codec=None):
        super().__init__(info_path, filters, transforms, codec)

    def load_meta(self, sample, info):
        raise NotImplementedError

    def load_data(self, sample, info):
        raise NotImplementedError

    def load_anno(self, sample, info):
        raise NotImplementedError

    def format(self, result, pred_path=None, gt_path=None):
        raise NotImplementedError

    def evaluate(self, predict_path, gt_path=None):
        raise NotImplementedError

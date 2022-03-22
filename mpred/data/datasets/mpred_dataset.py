from mai.data.datasets import BaseDataset


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

    def plot(self, sample, **kwargs):
        # TODO
        raise NotImplementedError

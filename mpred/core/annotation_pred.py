from mai.data.datasets import Annotation
import numpy as np


class AnnotationTrajPred(Annotation):
    def __init__(self, trajs, types=None, scores=None):
        super().__init__()

        r'''
        shape: K x L x 2, float32
        '''
        self.trajs = trajs

        r'''
        shape: K
        '''
        self.types = types if types is not None else np.full(
            trajs.shape[0], -1, dtype=np.int32)

        r'''
        shape: K
        '''
        self.scores = scores if scores is not None else np.full(
            trajs.shape[0], 0, dtype=trajs.dtype)

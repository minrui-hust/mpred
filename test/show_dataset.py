from mai.utils import FI
import mpred.data
import numpy as np

from enum import IntEnum

import open3d as o3d
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader

dataset = dict(
    type='ArgoPredDataset',
    info_path=f'/data/dataset/argoverse/prediction/val_info.pkl',
    load_opt=dict(
        traj_path=f'/data/dataset/argoverse/prediction/val_traj.npy',
        map_path=f'/data/dataset/argoverse/prediction/map_info.pkl',
        obs_len=20,
        pred_len=30,
        lane_radius=65,
    ),
    filters=[],
    transforms=[
        dict(type='MpredRetarget', prob=1.0, min_len=40, min_dist=1.0),
        #  dict(type='Normalize'),
        #  dict(type='MpredMaskHistory', mask_prob=0.8, max_len=10),
        #  dict(type='ObjectRangeFilter', obj_radius=10),
        #  dict(type='MpredMirrorFlip', mirror_prob=0, flip_prob=0.5),
        #  dict(type='MpredGlobalTransform', translation_std=[
        #       0, 0], rot_range=[-0.78539816, 0.78539816], scale_range=[0.5, 2.0]),
    ],
)

codec = dict(
    type='MMTransCodec',
    encode_cfg=dict(
        encode_data=True,
        encode_anno=True,
    ),
    decode_cfg=dict(),
    loss_cfg=dict(
        delta=1.8,
        wgt={
            'traj': 1.0,
            'score': 1.0,
        }
    ),
)


dataset = FI.create(dataset)
codec = FI.create(codec)
dataset.codec = codec

for i in range(len(dataset)):
    data = dataset[i]
    dataset.plot(data)

dataloader = DataLoader(dataset, batch_size=2,
                        num_workers=0, collate_fn=lambda x: x)

for batch in tqdm(dataloader):
    for data in batch:
        codec.plot(data)

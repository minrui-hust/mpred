from mai.utils import FI
import mpred.data
import numpy as np

from enum import IntEnum

import open3d as o3d
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader

dataset = dict(
    type='ArgoPredDataset',
    info_path=f'/data/dataset/argoverse/prediction/train_info.pkl',
    load_opt=dict(
        map_path=f'/data/dataset/argoverse/prediction/map_info.pkl',
        obs_len=20,
        pred_len=30,
    ),
    filters=[],
    transforms=[
        dict(type='Normlize'),
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
    #  dataset.plot(dataset[i])

dataloader = DataLoader(dataset, batch_size=2,
                        num_workers=0, collate_fn=codec.get_collater())

#  dataset.plot(dataset[0])
for batch in tqdm(dataloader):
    pass
    #  print(batch)

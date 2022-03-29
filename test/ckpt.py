import torch

#  ckpt = torch.load('log/centerpoint_pp_waymo_3cls_small_range_gtaug/version_0/epoch=35-step=355715.ckpt')
ckpt = torch.load('log/mmtrans_argoverse_base/version_0/epoch=16-step=20977.ckpt')
print(ckpt['state_dict'])
